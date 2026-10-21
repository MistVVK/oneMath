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

// The rocBLAS backend's GEMMs with 16-bit inputs and FP32 output, and with int8 inputs and int32 output, through
// hipBLASLt, where hipBLASLt has kernels for the GPU: rocBLAS has no tuned kernels for some GPUs (RDNA4's gfx12 in ROCm 7.1, where its generic ones ran at a fifteenth
// of hipBLASLt's speed), and AMD gives those GPUs' matrix products to hipBLASLt.  ONEMATH_HIPBLASLT=0 turns it off.

#ifndef _ROCBLAS_HIPBLASLT_HPP_
#define _ROCBLAS_HIPBLASLT_HPP_

#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <rocblas/rocblas.h>

#include <cstdint>

namespace oneapi {
namespace math {
namespace blas {
namespace rocblas {

// C (column-major, ldc) = alpha * op(A) * op(B) + beta * C on `stream`, A and B of type `type` (HIP_R_16F or
// HIP_R_16BF), C FP32, FP32 accumulation.  False when hipBLASLt does not take it (no kernel for this GPU or shape,
// or turned off): the caller runs rocBLAS then.
bool hipblaslt_gemm(hipStream_t stream, rocblas_operation transa, rocblas_operation transb, int64_t m,
                    int64_t n, int64_t k, float alpha, const void* a, int64_t lda, const void* b,
                    int64_t ldb, float beta, float* c, int64_t ldc, hipDataType type);

// The same for int8 A and B and int32 C with int32 accumulation and scaling, batch_size products whose matrices lie
// stridea, strideb and stridec elements apart: rocBLAS's own int8 kernels for some GPUs are generic ones (gfx12 in
// ROCm 7.1: a tenth of hipBLASLt's speed).
bool hipblaslt_gemm_i8(hipStream_t stream, rocblas_operation transa, rocblas_operation transb, int64_t m,
                       int64_t n, int64_t k, int32_t alpha, const void* a, int64_t lda, int64_t stridea,
                       const void* b, int64_t ldb, int64_t strideb, int32_t beta, int32_t* c,
                       int64_t ldc, int64_t stridec, int64_t batch_size);

} // namespace rocblas
} // namespace blas
} // namespace math
} // namespace oneapi

#endif // _ROCBLAS_HIPBLASLT_HPP_
