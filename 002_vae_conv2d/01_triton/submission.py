import torch
import torch.nn.functional as F
import triton
import triton.language as tl


GN_SILU_BLOCK = 1024


# ---------------------------------------------------------------------------
# Kernel 1 – partial stats (sum, sum_sq) per chunk, accumulated via atomics.
# Grid: (N * G * S,)
# ---------------------------------------------------------------------------
@triton.jit
def _gn_stats_kernel(
    x_ptr, stats_ptr,
    C, HW, G, C_per_G, group_size, chunk_size, S,
    BLOCK: tl.constexpr,
):
    pid      = tl.program_id(0)
    group_id = pid // S
    s        = pid  % S

    n = group_id // G
    g = group_id  % G
    group_start = n * C * HW + g * C_per_G * HW
    chunk_start = s * chunk_size

    _sum    = 0.0
    _sum_sq = 0.0
    for i in range(tl.cdiv(chunk_size, BLOCK)):
        offs = chunk_start + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < group_size
        x = tl.load(x_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)
        _sum    += tl.sum(x)
        _sum_sq += tl.sum(x * x)

    tl.atomic_add(stats_ptr + group_id * 2 + 0, _sum)
    tl.atomic_add(stats_ptr + group_id * 2 + 1, _sum_sq)


# ---------------------------------------------------------------------------
# Kernel 2a – read stats, normalize, SiLU.
# Grid: (N * G * S,)
# ---------------------------------------------------------------------------
@triton.jit
def _gn_norm_silu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr, stats_ptr,
    C, HW, G, C_per_G, group_size, chunk_size, S,
    eps,
    BLOCK: tl.constexpr,
):
    pid      = tl.program_id(0)
    group_id = pid // S
    s        = pid  % S

    n = group_id // G
    g = group_id  % G
    group_start = n * C * HW + g * C_per_G * HW
    chunk_start = s * chunk_size

    total_sum    = tl.load(stats_ptr + group_id * 2 + 0).to(tl.float32)
    total_sum_sq = tl.load(stats_ptr + group_id * 2 + 1).to(tl.float32)
    mean = total_sum    / group_size
    var  = total_sum_sq / group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    for i in range(tl.cdiv(chunk_size, BLOCK)):
        offs = chunk_start + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < group_size
        x = tl.load(x_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)

        c_local  = offs // HW
        c_global = g * C_per_G + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr   + c_global, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd * w + b
        y = y * tl.sigmoid(y)
        tl.store(out_ptr + group_start + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2b – read stats, normalize, SiLU, add residual.
# Grid: (N * G * S,)
# ---------------------------------------------------------------------------
@triton.jit
def _gn_norm_silu_add_kernel(
    x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr, stats_ptr,
    C, HW, G, C_per_G, group_size, chunk_size, S,
    eps,
    BLOCK: tl.constexpr,
):
    pid      = tl.program_id(0)
    group_id = pid // S
    s        = pid  % S

    n = group_id // G
    g = group_id  % G
    group_start = n * C * HW + g * C_per_G * HW
    chunk_start = s * chunk_size

    total_sum    = tl.load(stats_ptr + group_id * 2 + 0).to(tl.float32)
    total_sum_sq = tl.load(stats_ptr + group_id * 2 + 1).to(tl.float32)
    mean = total_sum    / group_size
    var  = total_sum_sq / group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    for i in range(tl.cdiv(chunk_size, BLOCK)):
        offs = chunk_start + i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < group_size
        x   = tl.load(x_ptr        + group_start + offs, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(residual_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)

        c_local  = offs // HW
        c_global = g * C_per_G + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr   + c_global, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd * w + b
        y = y * tl.sigmoid(y) + res
        tl.store(out_ptr + group_start + offs, y, mask=mask)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_supported_shape(x: torch.Tensor, num_groups: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"x must be 4D NCHW, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"x must be float32, got {x.dtype}")
    if x.shape[1] % num_groups != 0:
        raise ValueError(
            f"channels must be divisible by num_groups, got {x.shape[1]} and {num_groups}"
        )


def _compute_S(N: int, G: int, group_size: int, block: int) -> int:
    """
    Choose spatial split factor S so that N*G*S is large enough to saturate
    B200's SMs, but each chunk is at least `block` elements.
    """
    TARGET = 4096
    S = max(1, TARGET // (N * G))
    S = min(S, max(1, group_size // block))
    return S


# ---------------------------------------------------------------------------
# Python-side wrappers
# ---------------------------------------------------------------------------

def _group_norm_silu_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    _require_supported_shape(x, num_groups)
    x      = x.contiguous()
    weight = weight.contiguous()
    bias   = bias.contiguous()

    N, C, H, W = x.shape
    HW         = H * W
    C_per_G    = C // num_groups
    group_size = C_per_G * HW
    S          = _compute_S(N, num_groups, group_size, GN_SILU_BLOCK)
    chunk_size = triton.cdiv(group_size, S)
    grid       = (N * num_groups * S,)

    stats = torch.zeros(N * num_groups, 2, dtype=torch.float32, device=x.device)
    out   = torch.empty_like(x)

    _gn_stats_kernel[grid](
        x, stats,
        C, HW, num_groups, C_per_G, group_size, chunk_size, S,
        BLOCK=GN_SILU_BLOCK, num_warps=8,
    )
    _gn_norm_silu_kernel[grid](
        x, weight, bias, out, stats,
        C, HW, num_groups, C_per_G, group_size, chunk_size, S,
        eps,
        BLOCK=GN_SILU_BLOCK, num_warps=8,
    )
    return out


def _group_norm_silu_add_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    _require_supported_shape(x, num_groups)
    x        = x.contiguous()
    residual = residual.contiguous()
    weight   = weight.contiguous()
    bias     = bias.contiguous()

    N, C, H, W = x.shape
    HW         = H * W
    C_per_G    = C // num_groups
    group_size = C_per_G * HW
    S          = _compute_S(N, num_groups, group_size, GN_SILU_BLOCK)
    chunk_size = triton.cdiv(group_size, S)
    grid       = (N * num_groups * S,)

    stats = torch.zeros(N * num_groups, 2, dtype=torch.float32, device=x.device)
    out   = torch.empty_like(x)

    _gn_stats_kernel[grid](
        x, stats,
        C, HW, num_groups, C_per_G, group_size, chunk_size, S,
        BLOCK=GN_SILU_BLOCK, num_warps=8,
    )
    _gn_norm_silu_add_kernel[grid](
        x, weight, bias, residual, out, stats,
        C, HW, num_groups, C_per_G, group_size, chunk_size, S,
        eps,
        BLOCK=GN_SILU_BLOCK, num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    num_groups = 32
    """
    Fused residual block: Conv3x3 -> [GroupNorm+SiLU] -> Conv3x3 -> [GroupNorm+SiLU+Add].

    Each [GroupNorm+SiLU] is split into two Triton kernels:
      1. _gn_stats_kernel:    N*G*S programs compute partial sum/sum_sq via atomics.
      2. _gn_norm_silu_kernel: N*G*S programs read completed stats, normalize, activate.
    Splitting each group into S spatial chunks gives enough parallelism to
    saturate B200's SMs even at small batch sizes.
    """
    _require_supported_shape(x, num_groups)
    if not x.is_cuda:
        raise ValueError("run requires CUDA tensors")

    kernel_size = conv1_weight.shape[-1]
    if conv1_weight.shape[-2] != kernel_size or conv2_weight.shape[-1] != kernel_size:
        raise ValueError("conv weights must be square and share the same kernel size")
    padding  = kernel_size // 2
    residual = x.contiguous()

    out = F.conv2d(residual, conv1_weight.contiguous(), bias=None, stride=1, padding=padding)
    out = _group_norm_silu_triton(out, norm1_weight, norm1_bias, eps, num_groups)

    out = F.conv2d(out, conv2_weight.contiguous(), bias=None, stride=1, padding=padding)
    out = _group_norm_silu_add_triton(out, norm2_weight, norm2_bias, residual, eps, num_groups)
    return out
