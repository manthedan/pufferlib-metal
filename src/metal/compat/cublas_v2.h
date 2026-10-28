// Metal backend: cuBLAS stand-in. Only the GEMM entry point PufferLib uses
// (cublasGemmEx, column-major, FP32) is implemented, in src/metal/gemm.mm.
#pragma once
#include "cuda_runtime.h"

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
} cublasStatus_t;
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;
typedef enum { CUBLAS_COMPUTE_32F = 68 } cublasComputeType_t;
typedef enum { CUBLAS_GEMM_DEFAULT = -1 } cublasGemmAlgo_t;
typedef enum { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TF32_TENSOR_OP_MATH = 3 } cublasMath_t;
typedef struct cublasContext* cublasHandle_t;

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* ws, size_t bytes);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasGemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha,
    const void* A, cudaDataType_t Atype, int lda,
    const void* B, cudaDataType_t Btype, int ldb,
    const void* beta, void* C, cudaDataType_t Ctype, int ldc,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo);
