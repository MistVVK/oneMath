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

// what a plan depends on: the operations, the inputs' type, the shape, the leading dimensions, the batch and its strides
using Key = std::tuple<int, int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       int64_t, int64_t>;

// the types of a product: the inputs', the output's, the accumulation's and the scales'
struct Types {
    hipDataType in, out;
    hipblasComputeType_t compute;
    hipDataType scale;
};

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

// a layout of rows x cols with leading dimension ld, and batch matrices stride elements apart
hipblasLtMatrixLayout_t layout(hipDataType type, int64_t rows, int64_t cols, int64_t ld, int64_t batch,
                               int64_t stride) {
    hipblasLtMatrixLayout_t l = nullptr;
    if (hipblasLtMatrixLayoutCreate(&l, type, rows, cols, ld) != HIPBLAS_STATUS_SUCCESS)
        return nullptr;
    if (batch > 1) {
        const int32_t count = (int32_t)batch;
        hipblasLtMatrixLayoutSetAttribute(l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof count);
        hipblasLtMatrixLayoutSetAttribute(l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                                          sizeof stride);
    }
    return l;
}

Plan make_plan(hipblasLtHandle_t handle, rocblas_operation transa, rocblas_operation transb, int64_t m,
               int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc, const Types& t, int64_t batch,
               int64_t stridea, int64_t strideb, int64_t stridec) {
    Plan p;
    if (hipblasLtMatmulDescCreate(&p.desc, t.compute, t.scale) != HIPBLAS_STATUS_SUCCESS)
        return p;
    const hipblasOperation_t ta = op(transa), tb = op(transb);
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
    hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
    // the stored matrices: A is m x k (k x m transposed), B k x n (n x k transposed), column-major
    p.la = layout(t.in, ta == HIPBLAS_OP_N ? m : k, ta == HIPBLAS_OP_N ? k : m, lda, batch, stridea);
    p.lb = layout(t.in, tb == HIPBLAS_OP_N ? k : n, tb == HIPBLAS_OP_N ? n : k, ldb, batch, strideb);
    p.lc = layout(t.out, m, n, ldc, batch, stridec);
    if (p.la == nullptr || p.lb == nullptr || p.lc == nullptr)
        return p;
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

// C = alpha * op(A) * op(B) + beta * C through the plan for this product, alpha and beta of t.scale
bool run(hipStream_t stream, rocblas_operation transa, rocblas_operation transb, int64_t m, int64_t n, int64_t k,
         const void* alpha, const void* a, int64_t lda, int64_t stridea, const void* b, int64_t ldb,
         int64_t strideb, const void* beta, void* c, int64_t ldc, int64_t stridec, int64_t batch,
         const Types& t) {
    if (turned_off() || m <= 0 || n <= 0 || k <= 0 || batch <= 0)
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
        if (batch == 1)
            stridea = strideb = stridec = 0;
        const Key key{(int)transa, (int)transb, (int)t.in, m, n, k, lda, ldb, ldc, batch, stridea, strideb, stridec};
        auto it = d->plans.find(key);
        if (it == d->plans.end())
            it = d->plans
                     .emplace(key, make_plan(d->handle, transa, transb, m, n, k, lda, ldb, ldc, t, batch,
                                             stridea, strideb, stridec))
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
    return hipblasLtMatmul(handle, plan.desc, alpha, a, plan.la, b, plan.lb, beta, c, plan.lc, c, plan.lc,
                           &plan.algo, workspace, kWorkspace, stream) == HIPBLAS_STATUS_SUCCESS;
}

} // namespace

bool hipblaslt_gemm(hipStream_t stream, rocblas_operation transa, rocblas_operation transb, int64_t m,
                    int64_t n, int64_t k, float alpha, const void* a, int64_t lda, const void* b,
                    int64_t ldb, float beta, float* c, int64_t ldc, hipDataType type) {
    return run(stream, transa, transb, m, n, k, &alpha, a, lda, 0, b, ldb, 0, &beta, c, ldc, 0, 1,
               Types{type, HIP_R_32F, HIPBLAS_COMPUTE_32F, HIP_R_32F});
}

bool hipblaslt_gemm_i8(hipStream_t stream, rocblas_operation transa, rocblas_operation transb, int64_t m,
                       int64_t n, int64_t k, int32_t alpha, const void* a, int64_t lda, int64_t stridea,
                       const void* b, int64_t ldb, int64_t strideb, int32_t beta, int32_t* c,
                       int64_t ldc, int64_t stridec, int64_t batch_size) {
    return run(stream, transa, transb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc,
               stridec, batch_size, Types{HIP_R_8I, HIP_R_32I, HIPBLAS_COMPUTE_32I, HIP_R_32I});
}

} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi
