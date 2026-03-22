#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <stdexcept>
#include <string>

#define CUBLASLT_CHECK(expr)                                                  \
    do {                                                                      \
        cublasStatus_t _s = (expr);                                           \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(                                         \
                std::string("cublasLt error: ") + std::to_string((int)_s));  \
        }                                                                     \
    } while (0)

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            throw std::runtime_error(                                         \
                std::string("CUDA error: ") + cudaGetErrorString(_e));       \
        }                                                                     \
    } while (0)

/*
 * Fused GEMM + residual add (bfloat16):
 *   output = attn_output @ o_proj_weight.T + residual
 *
 * attn_output  : (..., K)  bfloat16  — M = numel / K
 * o_proj_weight: (N, K)   bfloat16
 * residual     : (..., N)  bfloat16
 *
 * Uses cublasLtMatmul with alpha=1, beta=1 so the residual add is
 * folded into the GEMM epilogue (single global-memory write for C).
 * All layouts are explicitly set to row-major (CUBLASLT_ORDER_ROW).
 */
torch::Tensor run(
    const torch::Tensor& attn_output,
    const torch::Tensor& residual,
    const torch::Tensor& o_proj_weight)
{
    TORCH_CHECK(attn_output.is_cuda(),   "attn_output must be a CUDA tensor");
    TORCH_CHECK(residual.is_cuda(),      "residual must be a CUDA tensor");
    TORCH_CHECK(o_proj_weight.is_cuda(), "o_proj_weight must be a CUDA tensor");
    TORCH_CHECK(attn_output.scalar_type() == torch::kBFloat16, "inputs must be bfloat16");

    const auto orig_shape = attn_output.sizes().vec();
    const int64_t M = attn_output.numel() / attn_output.size(-1);
    const int64_t K = attn_output.size(-1);
    const int64_t N = o_proj_weight.size(0);

    auto A      = attn_output.contiguous().view({M, K});
    auto B      = o_proj_weight.contiguous();        // (N, K)
    auto R      = residual.contiguous().view({M, N});
    auto output = torch::empty({M, N}, A.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cublasLtHandle_t ltHandle;
    CUBLASLT_CHECK(cublasLtCreate(&ltHandle));

    // ----------------------------------------------------------------
    // MatMul descriptor: BF16 tensor core, F32 accumulation
    // ----------------------------------------------------------------
    cublasLtMatmulDesc_t matmulDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(
        &matmulDesc, CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F));

    // Col-major reformulation (no row-major constraint, full algorithm set):
    //   want (row-major):  D = A @ B^T + R
    //   equiv (col-major): D^T = B @ A^T + R^T
    //   A_row(M×K) = col-major K×M (ld=K)
    //   B_row(N×K) = col-major K×N (ld=K), pass as A' with opA=T → N×K
    //   R/D row-major(M×N) = col-major N×M (ld=N)
    cublasOperation_t opA = CUBLAS_OP_T;  // B_row (K×N col-major) → N×K
    cublasOperation_t opB = CUBLAS_OP_N;  // A_row (K×M col-major) stays K×M
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    // ----------------------------------------------------------------
    // Matrix layouts — native col-major, no ORDER_ROW constraint
    // ----------------------------------------------------------------
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;
    // A' = B_row viewed as K×N col-major, ld=K
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, K, N, K));
    // B' = A_row viewed as K×M col-major, ld=K
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, K, M, K));
    // C = residual viewed as N×M col-major, ld=N
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, N, M, N));
    // D = output viewed as N×M col-major, ld=N
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_16BF, N, M, N));

    // ----------------------------------------------------------------
    // Algorithm heuristic search
    // ----------------------------------------------------------------
    cublasLtMatmulPreference_t pref;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t maxWs = 32ULL * 1024 * 1024;  // 32 MiB workspace budget
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs)));

    int returnedAlgoCount = 0;
    cublasLtMatmulHeuristicResult_t heurResult;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc,
        layoutA, layoutB, layoutC, layoutD,
        pref, 1, &heurResult, &returnedAlgoCount));
    TORCH_CHECK(returnedAlgoCount > 0, "cublasLt: no suitable algorithm found");

    // ----------------------------------------------------------------
    // Workspace allocation (may be 0 bytes)
    // ----------------------------------------------------------------
    void* workspace = nullptr;
    if (heurResult.workspaceSize > 0)
        CUDA_CHECK(cudaMalloc(&workspace, heurResult.workspaceSize));

    // ----------------------------------------------------------------
    // Execute: D^T = 1·op(B_row)·A_row^T + 1·R^T  (col-major, fused residual)
    // ----------------------------------------------------------------
    const float alpha = 1.0f, beta = 1.0f;
    CUBLASLT_CHECK(cublasLtMatmul(
        ltHandle, matmulDesc,
        &alpha,
        B.data_ptr(), layoutA,   // A' ← B_row (K×N col-major, transposed to N×K)
        A.data_ptr(), layoutB,   // B' ← A_row (K×M col-major)
        &beta,
        R.data_ptr(), layoutC,
        output.data_ptr(), layoutD,
        &heurResult.algo,
        workspace, heurResult.workspaceSize,
        stream));

    // ----------------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------------
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutD));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLASLT_CHECK(cublasLtMatmulDescDestroy(matmulDesc));
    CUBLASLT_CHECK(cublasLtDestroy(ltHandle));

    // Restore original batch dims with N as last dim
    auto out_shape = orig_shape;
    out_shape.back() = N;
    return output.view(out_shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run,
          "cublasLt fused GEMM + residual add (bfloat16): "
          "output = attn_output @ o_proj_weight.T + residual");
}
