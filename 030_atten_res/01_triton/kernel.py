import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel: fused o_proj matmul + residual add  (TMA edition)
#
#   C[m, n] = sum_k A[m, k] * B[n, k]  +  R[m, n]
#
# A = attn_output reshaped to (M, K),  M = batch * seq_len, K = hidden_size
# B = o_proj_weight of shape (N, K),   N = hidden_size  (accessed transposed)
# R = residual reshaped to (M, N)
# C = output reshaped to (M, N)
#
# TMA descriptors are created once per CTA at kernel entry.
# Hardware bounds-checking removes the need for explicit masks.
#
# Requires Hopper / Blackwell (sm_90+).
#
# Grid: (ceil(M / BLOCK_M),  ceil(N / BLOCK_N))
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ---- small tiles ----
        triton.Config({"BLOCK_M":  64, "BLOCK_N":  64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M":  64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N":  64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        # ---- medium tiles ----
        triton.Config({"BLOCK_M":  64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N":  64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        # ---- large tiles (B200: 228 KB SRAM, deep pipeline) ----
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=16, num_stages=5),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_matmul_residual_tma_kernel(
    # Pointers
    A_ptr, B_ptr, R_ptr, C_ptr,
    # Dimensions
    M, N, K,
    # Row strides (innermost stride is always 1 for row-major contiguous tensors)
    stride_am,   # A: (M, K) -> row stride = K
    stride_bn,   # B: (N, K) -> row stride = K
    stride_rm,   # R: (M, N) -> row stride = N
    stride_cm,   # C: (M, N) -> row stride = N
    # Tile sizes (constexpr for TMA block_shape)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Each CTA computes a (BLOCK_M, BLOCK_N) output tile:
        C[m_tile, n_tile] = A[m_tile, :] @ B[n_tile, :]^T  +  R[m_tile, n_tile]

    TMA loads supply entire tiles asynchronously; hardware handles bounds.
    The matmul accumulates in float32; the residual add and store are bf16.
    """
    # ---- Build TMA descriptors (once per CTA) ------------------------------
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, K],
        strides=[stride_bn, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    r_desc = tl.make_tensor_descriptor(
        R_ptr,
        shape=[M, N],
        strides=[stride_rm, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N],
        strides=[stride_cm, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # ---- K-loop: accumulate matmul tiles ------------------------------------
    for k_start in tl.range(0, K, BLOCK_K):
        # TMA load A tile: [BLOCK_M, BLOCK_K]
        A_tile = tl.load_tensor_descriptor(a_desc, [m_start, k_start])

        # TMA load B tile: [BLOCK_N, BLOCK_K]  (row = output feature)
        B_tile = tl.load_tensor_descriptor(b_desc, [n_start, k_start])

        # A[BLOCK_M, BLOCK_K] @ B^T[BLOCK_K, BLOCK_N] -> accumulate [BLOCK_M, BLOCK_N]
        acc += tl.dot(A_tile, tl.trans(B_tile))  # fp32 accumulation

    # ---- Fused residual add and store ---------------------------------------
    R_tile = tl.load_tensor_descriptor(r_desc, [m_start, n_start])  # [BLOCK_M, BLOCK_N]

    out = (acc + R_tile.to(tl.float32)).to(tl.bfloat16)

    tl.store_tensor_descriptor(c_desc, [m_start, n_start], out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Fused Triton (TMA) implementation of attention output projection + residual add.

    Performs:  output = attn_output @ o_proj_weight.T + residual

    All tensors must be bfloat16, contiguous, on a Hopper/Blackwell CUDA device.
    """
    B, S, H_in = attn_output.shape
    H_out = o_proj_weight.shape[0]

    M = B * S
    N = H_out
    K = H_in

    # Flatten batch + seq dims for 2-D GEMM; ensure contiguous for TMA
    A = attn_output.reshape(M, K).contiguous()   # (M, K)
    R = residual.reshape(M, N).contiguous()       # (M, N)
    W = o_proj_weight.contiguous()                # (N, K)

    C = torch.empty(M, N, dtype=torch.bfloat16, device=attn_output.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _fused_matmul_residual_tma_kernel[grid](
        A, W, R, C,
        M, N, K,
        A.stride(0),   # stride_am
        W.stride(0),   # stride_bn
        R.stride(0),   # stride_rm
        C.stride(0),   # stride_cm
    )

    return C.view(B, S, N)
