/***************************************************************************
*  Copyright (C) 2026 MistVVK and the XeStrata contributors
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#include "rocblas_hipblaslt.hpp"

#include <hipblaslt/hipblaslt.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace oneapi {
namespace math {
namespace blas {
namespace rocblas {

namespace {

// hipBLASLt's workspace, which the kernels that split K need (without one the heuristic leaves them out); 32 MiB, as
// Strata's HIP engine gives it
constexpr size_t kWorkspace = size_t{32} << 20;

// A shape's descriptors and the algorithm the heuristic picked for it (none: hipBLASLt has no kernel for it)
struct Plan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    bool ok = false;
};

using Key = std::tuple<int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

// A GPU's handle, plans and a workspace for each stream (streams run at once must not share one); a GPU where the
// handle cannot be made answers no once and is not asked again
struct Device {
    hipblasLtHandle_t handle = nullptr;
    bool usable = false;
    std::map<Key, Plan> plans;
    std::map<hipStream_t, void*> workspaces;
};

std::mutex g_mutex;
std::map<int, std::unique_ptr<Device>> g_devices;

bool turned_off() {
    static const bool off = [] {
        const char* v = std::getenv("ONEMATH_HIPBLASLT");
        return v != nullptr && std::strcmp(v, "0") == 0;
    }();
    return off;
}

Device* device_state(int dev) {
    auto& d = g_devices[dev];
    if (!d) {
        d = std::make_unique<Device>();
        d->usable = hipblasLtCreate(&d->handle) == HIPBLAS_STATUS_SUCCESS;
    }
    return d->usable ? d.get() : nullptr;
}

hipblasOperation_t op(rocblas_operation o) {
    return o == rocblas_operation_none ? HIPBLAS_OP_N
                                       : (o == rocblas_operation_transpose ? HIPBLAS_OP_T : HIPBLAS_OP_C);
}

Plan make_plan(hipblasLtHandle_t handle, rocblas_operation transa, rocblas_operation transb, int64_t m,
               int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc, hipDataType type) {
    Plan p;
    if (hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS)
        return p;
    const hipblasOperation_t ta = op(transa), tb = op(transb);
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
    // the stored matrices: A is m x k (k x m transposed), B k x n (n x k transposed), column-major
    hipblasLtMatrixLayoutCreate(&p.la, type, ta == HIPBLAS_OP_N ? m : k, ta == HIPBLAS_OP_N ? k : m, lda);
    hipblasLtMatrixLayoutCreate(&p.lb, type, tb == HIPBLAS_OP_N ? k : n, tb == HIPBLAS_OP_N ? n : k, ldb);
    hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_32F, m, n, ldc);
    hipblasLtMatmulPreference_t pref;
    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS)
        return p;
    size_t ws = kWorkspace;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws);
    hipblasLtMatmulHeuristicResult_t r{};
    int found = 0;
    if (hipblasLtMatmulAlgoGetHeuristic(handle, p.desc, p.la, p.lb, p.lc, p.lc, pref, 1, &r, &found) ==
            HIPBLAS_STATUS_SUCCESS &&
        found > 0) {
        p.algo = r.algo;
        p.ok = true;
    }
    hipblasLtMatmulPreferenceDestroy(pref);
    return p;
}

} // namespace

bool hipblaslt_gemm(hipStream_t stream, rocblas_operation transa, rocblas_operation transb, int64_t m,
                    int64_t n, int64_t k, float alpha, const void* a, int64_t lda, const void* b,
                    int64_t ldb, float beta, float* c, int64_t ldc, hipDataType type) {
    if (turned_off() || m <= 0 || n <= 0 || k <= 0)
        return false;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess)
        return false;
    hipblasLtHandle_t handle;
    void* workspace;
    Plan plan;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        Device* d = device_state(dev);
        if (d == nullptr)
            return false;
        const Key key{(int)transa, (int)transb, (int)type, m, n, k, lda, ldb, ldc};
        auto it = d->plans.find(key);
        if (it == d->plans.end())
            it = d->plans.emplace(key, make_plan(d->handle, transa, transb, m, n, k, lda, ldb, ldc, type))
                     .first;
        if (!it->second.ok)
            return false;
        void*& ws = d->workspaces[stream];
        if (ws == nullptr && hipMalloc(&ws, kWorkspace) != hipSuccess) {
            ws = nullptr;
            return false;
        }
        handle = d->handle;
        workspace = ws;
        plan = it->second;
    }
    return hipblasLtMatmul(handle, plan.desc, &alpha, a, plan.la, b, plan.lb, &beta, c, plan.lc, c, plan.lc,
                           &plan.algo, workspace, kWorkspace, stream) == HIPBLAS_STATUS_SUCCESS;
}

} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi
