import torch


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Fused attention output projection with residual addition.

    This performs:
    1. Linear projection: attn_output @ o_proj_weight.T
    2. Residual addition: projected + residual

    In a custom CUDA kernel, these operations would be fused to:
    - Compute matmul tiles in registers
    - Add residual directly to register-held results
    - Write final result to global memory (eliminating intermediate write)

    Args:
        attn_output: Attention output of shape (batch, seq_len, hidden_size)
        residual: Original input before attention of shape (batch, seq_len, hidden_size)
        o_proj_weight: Output projection weight of shape (hidden_size, hidden_size)

    Returns:
        Output with residual added, shape (batch, seq_len, hidden_size)
    """
    # Linear projection: (batch, seq_len, hidden_size) @ (hidden_size, hidden_size).T
    # -> (batch, seq_len, hidden_size)
    projected = torch.matmul(attn_output, o_proj_weight.t())

    # Residual addition
    output = projected + residual

    return output
