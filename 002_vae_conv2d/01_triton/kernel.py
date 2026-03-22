import torch
import torch.nn.functional as F
import triton
import triton.language as tl


GN_SILU_BLOCK  = 1024
NUM_SMS        = 160   # B200
TARGET_PROGRAMS = NUM_SMS * 8   # 1280: enough waves to hide scheduling


# ===========================================================================
# Path A — one program per group (used when N*G >= NUM_SMS).
# Stats and normalize are fused: stats accumulate in registers, never touch
# global memory.
# Grid: (N * G,)
# ===========================================================================

@triton.jit
def _gn_fused_norm_silu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW, G, C_per_G, group_size,
    eps,
    BLOCK: tl.constexpr,
):
    group_id    = tl.program_id(0)
    n           = group_id // G
    g           = group_id  % G
    group_start = n * C * HW + g * C_per_G * HW

    # Pass 1: accumulate sum / sum_sq in registers.
    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    _ssq = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(tl.cdiv(group_size, BLOCK)):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < group_size
        x    = tl.load(x_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)
        _sum += x
        _ssq += x * x

    mean = tl.sum(_sum) / group_size
    var  = tl.sum(_ssq) / group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Pass 2: normalize + SiLU.
    for i in range(tl.cdiv(group_size, BLOCK)):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < group_size
        x = tl.load(x_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)

        c_local  = offs // HW
        c_global = g * C_per_G + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr   + c_global, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd * w + b
        y = y * tl.sigmoid(y)
        tl.store(out_ptr + group_start + offs, y, mask=mask)


@triton.jit
def _gn_fused_norm_silu_add_kernel(
    x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr,
    C, HW, G, C_per_G, group_size,
    eps,
    BLOCK: tl.constexpr,
):
    group_id    = tl.program_id(0)
    n           = group_id // G
    g           = group_id  % G
    group_start = n * C * HW + g * C_per_G * HW

    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    _ssq = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(tl.cdiv(group_size, BLOCK)):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < group_size
        x    = tl.load(x_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)
        _sum += x
        _ssq += x * x

    mean = tl.sum(_sum) / group_size
    var  = tl.sum(_ssq) / group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    for i in range(tl.cdiv(group_size, BLOCK)):
        offs = i * BLOCK + tl.arange(0, BLOCK)
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


# ===========================================================================
# Path B — spatial split, no atomics (used when N*G < NUM_SMS).
#
# Kernel 1 (_gn_partial_stats_kernel):
#   Grid (N*G*S,).  Each program writes its partial sum/sum_sq to its own
#   unique slot in partial[pid*2 : pid*2+2] — no collision, no atomic.
#
# Kernel 2 (_gn_split_norm_silu[_add]_kernel):
#   Grid (N*G*S,).  Each program reads all S partial entries for its group
#   (O(S) scalar loads, S is small), reduces inline to get mean/rstd, then
#   normalizes its spatial chunk.
#
# The implicit barrier between the two launches ensures all partial stats
# are committed before any norm program reads them.
# ===========================================================================

@triton.jit
def _gn_partial_stats_kernel(
    x_ptr, partial_ptr,
    C, HW, G, C_per_G, group_size, chunk_size, S,
    BLOCK: tl.constexpr,
):
    pid         = tl.program_id(0)
    group_id    = pid // S
    s           = pid  % S
    n           = group_id // G
    g           = group_id  % G
    group_start = n * C * HW + g * C_per_G * HW
    chunk_start = s * chunk_size

    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    _ssq = tl.zeros([BLOCK], dtype=tl.float32)
    chunk_end = chunk_start + chunk_size
    for i in range(tl.cdiv(chunk_size, BLOCK)):
        offs = chunk_start + i * BLOCK + tl.arange(0, BLOCK)
        mask = (offs < chunk_end) & (offs < group_size)
        x    = tl.load(x_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)
        _sum += x
        _ssq += x * x

    # Unique slot per pid — no atomic.
    tl.store(partial_ptr + pid * 2 + 0, tl.sum(_sum))
    tl.store(partial_ptr + pid * 2 + 1, tl.sum(_ssq))


@triton.jit
def _gn_split_norm_silu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr, partial_ptr,
    C, HW, G, C_per_G, group_size, chunk_size, S,
    eps,
    BLOCK: tl.constexpr,
):
    pid         = tl.program_id(0)
    group_id    = pid // S
    s           = pid  % S
    n           = group_id // G
    g           = group_id  % G
    group_start = n * C * HW + g * C_per_G * HW
    chunk_start = s * chunk_size

    # Inline reduce: O(S) scalar reads to get full-group stats.
    total_sum = 0.0
    total_ssq = 0.0
    base = group_id * S
    for i in range(S):
        total_sum += tl.load(partial_ptr + (base + i) * 2 + 0).to(tl.float32)
        total_ssq += tl.load(partial_ptr + (base + i) * 2 + 1).to(tl.float32)

    mean = total_sum / group_size
    var  = total_ssq / group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    chunk_end = chunk_start + chunk_size
    for i in range(tl.cdiv(chunk_size, BLOCK)):
        offs = chunk_start + i * BLOCK + tl.arange(0, BLOCK)
        mask = (offs < chunk_end) & (offs < group_size)
        x = tl.load(x_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)

        c_local  = offs // HW
        c_global = g * C_per_G + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr   + c_global, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd * w + b
        y = y * tl.sigmoid(y)
        tl.store(out_ptr + group_start + offs, y, mask=mask)


@triton.jit
def _gn_split_norm_silu_add_kernel(
    x_ptr, weight_ptr, bias_ptr, residual_ptr, out_ptr, partial_ptr,
    C, HW, G, C_per_G, group_size, chunk_size, S,
    eps,
    BLOCK: tl.constexpr,
):
    pid         = tl.program_id(0)
    group_id    = pid // S
    s           = pid  % S
    n           = group_id // G
    g           = group_id  % G
    group_start = n * C * HW + g * C_per_G * HW
    chunk_start = s * chunk_size

    total_sum = 0.0
    total_ssq = 0.0
    base = group_id * S
    for i in range(S):
        total_sum += tl.load(partial_ptr + (base + i) * 2 + 0).to(tl.float32)
        total_ssq += tl.load(partial_ptr + (base + i) * 2 + 1).to(tl.float32)

    mean = total_sum / group_size
    var  = total_ssq / group_size - mean * mean
    rstd = tl.rsqrt(var + eps)

    chunk_end = chunk_start + chunk_size
    for i in range(tl.cdiv(chunk_size, BLOCK)):
        offs = chunk_start + i * BLOCK + tl.arange(0, BLOCK)
        mask = (offs < chunk_end) & (offs < group_size)
        x   = tl.load(x_ptr        + group_start + offs, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(residual_ptr + group_start + offs, mask=mask, other=0.0).to(tl.float32)

        c_local  = offs // HW
        c_global = g * C_per_G + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr   + c_global, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd * w + b
        y = y * tl.sigmoid(y) + res
        tl.store(out_ptr + group_start + offs, y, mask=mask)


# ===========================================================================
# Helpers
# ===========================================================================

def _require_supported_shape(x: torch.Tensor, num_groups: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"x must be 4D NCHW, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"x must be float32, got {x.dtype}")
    if x.shape[1] % num_groups != 0:
        raise ValueError(
            f"channels must be divisible by num_groups, got {x.shape[1]} and {num_groups}"
        )


def _launch_gn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual,   # torch.Tensor | None
    out: torch.Tensor,
    num_groups: int,
    eps: float,
) -> None:
    N, C, H, W = x.shape
    HW         = H * W
    C_per_G    = C // num_groups
    group_size = C_per_G * HW
    NG         = N * num_groups

    if NG >= NUM_SMS:
        # Path A: one fused program per group.
        grid = (NG,)
        kw   = dict(BLOCK=GN_SILU_BLOCK, num_warps=16, num_stages=4)
        if residual is None:
            _gn_fused_norm_silu_kernel[grid](
                x, weight, bias, out,
                C, HW, num_groups, C_per_G, group_size, eps, **kw,
            )
        else:
            _gn_fused_norm_silu_add_kernel[grid](
                x, weight, bias, residual, out,
                C, HW, num_groups, C_per_G, group_size, eps, **kw,
            )
    else:
        # Path B: spatial split — no atomics.
        S          = max(2, TARGET_PROGRAMS // NG)
        chunk_size = triton.cdiv(group_size, S)
        grid       = (NG * S,)
        partial    = torch.empty(NG * S, 2, dtype=torch.float32, device=x.device)
        kw         = dict(BLOCK=GN_SILU_BLOCK, num_warps=16, num_stages=4)

        _gn_partial_stats_kernel[grid](
            x, partial,
            C, HW, num_groups, C_per_G, group_size, chunk_size, S, **kw,
        )
        if residual is None:
            _gn_split_norm_silu_kernel[grid](
                x, weight, bias, out, partial,
                C, HW, num_groups, C_per_G, group_size, chunk_size, S, eps, **kw,
            )
        else:
            _gn_split_norm_silu_add_kernel[grid](
                x, weight, bias, residual, out, partial,
                C, HW, num_groups, C_per_G, group_size, chunk_size, S, eps, **kw,
            )


# ===========================================================================
# Python-side wrappers
# ===========================================================================

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
    out    = torch.empty_like(x)
    _launch_gn(x, weight, bias, None, out, num_groups, eps)
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
    out      = torch.empty_like(x)
    _launch_gn(x, weight, bias, residual, out, num_groups, eps)
    return out


# ===========================================================================
# Entry point
# ===========================================================================

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

    GroupNorm dispatch:
      Path A (N*G >= 160): single fused kernel per group; stats stay in registers.
      Path B (N*G <  160): spatial-split into S chunks; partial stats written to
                            unique per-chunk slots (no atomics), reduced inline in
                            the norm kernel before normalizing each chunk.
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
