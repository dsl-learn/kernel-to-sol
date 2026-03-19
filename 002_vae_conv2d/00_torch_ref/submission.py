import torch
import torch.nn.functional as F

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
    """
    Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add

    Args:
        x: Input tensor of shape (B, C, H, W)
        conv1_weight: First conv weights (C, C, 3, 3)
        norm1_weight: First GroupNorm scale (C,)
        norm1_bias: First GroupNorm bias (C,)
        conv2_weight: Second conv weights (C, C, 3, 3)
        norm2_weight: Second GroupNorm scale (C,)
        norm2_bias: Second GroupNorm bias (C,)
        eps: Epsilon for GroupNorm numerical stability

    Returns:
        Output tensor of shape (B, C, H, W)
    """
    num_groups = 32

    # Save residual
    residual = x

    # First path: Conv3x3 -> GroupNorm -> SiLU
    out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
    out = F.silu(out)

    # Second path: Conv3x3 -> GroupNorm -> SiLU
    out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
    out = F.silu(out)

    # Residual connection
    out = out + residual

    return out
