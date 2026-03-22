import torch
import cuda.tile as ct
from math import ceil

ConstInt = ct.Constant[int]

# Fixed shapes: M = 16 * 512 = 8192, N = K = 2560
# Tile sizes chosen so all dims divide exactly — no padding needed.
#   tm=128: M/tm = 64
#   tn=256: N/tn = 10   →  640 CTAs total (64 × 10)
#   tk=64:  K/tk = 40   →  40 K-loop iterations per CTA
TM, TN, TK = 128, 256, 64


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    """L2-friendly CTA swizzle (same pattern as sol-execbench reference)."""
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def fused_matmul_residual_kernel(
    A, B, R, C,
    tm: ConstInt, tn: ConstInt, tk: ConstInt,
):
    """
    Fused kernel:  C = A @ B + R   (all bfloat16)

    A : (M, K)  — attn_output flattened          M=8192, K=2560
    B : (K, N)  — o_proj_weight.T contiguous             N=2560
    R : (M, N)  — residual flattened
    C : (M, N)  — output

    Shapes always divisible by tile sizes → no padding_mode needed.
    Accumulator stays in registers; R tile is loaded once at the end
    and fused into the single global-memory write.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]

    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.float32)

    # bfloat16 uses native bf16 MMA — no tfloat32 cast needed
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))
        b = ct.load(B, index=(k, bidy), shape=(tk, tn))
        acc = ct.mma(a, b, acc)

    # Fused residual add: accumulator in registers, one store
    r = ct.load(R, index=(bidx, bidy), shape=(tm, tn))
    ct.store(C, index=(bidx, bidy), tile=ct.astype(acc, C.dtype) + r)


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    shape = attn_output.shape
    M = attn_output.numel() // attn_output.shape[-1]
    K, N = attn_output.shape[-1], o_proj_weight.shape[0]

    A = attn_output.contiguous().view(M, K)
    B = o_proj_weight.t().contiguous()          # (K, N)
    R = residual.contiguous().view(M, N)
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    grid = (ceil(M / TM) * ceil(N / TN), 1, 1)   # = 640

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_matmul_residual_kernel,
        (A, B, R, C, TM, TN, TK),
    )

    return C.view(shape)
