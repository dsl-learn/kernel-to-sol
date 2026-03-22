import torch


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for backward pass testing."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len_q = axes_and_scalars["seq_len_q"]
    seq_len_kv = axes_and_scalars["seq_len_kv"]
    num_attention_heads = 80
    num_key_value_heads = 8
    head_dim = 128
    # Use a fixed dropout probability for testing
    attention_dropout = 0.1

    # Gradient of attention output
    grad_attn_output = torch.randn(
        batch_size, seq_len_q, num_attention_heads, head_dim,
        dtype=torch.bfloat16, device=device
    )

    # Attention weights after softmax (should sum to 1 along last dim)
    attn_scores_raw = torch.randn(
        batch_size, num_attention_heads, seq_len_q, seq_len_kv,
        dtype=torch.float32, device=device
    )
    attn_weights = torch.softmax(attn_scores_raw, dim=-1).to(torch.bfloat16)

    # Generate dropout mask
    dropout_mask = torch.rand(
        batch_size, num_attention_heads, seq_len_q, seq_len_kv,
        device=device
    ) > attention_dropout

    # Attention weights after dropout
    if attention_dropout > 0.0:
        attn_weights_dropped = (attn_weights.float() * dropout_mask / (1.0 - attention_dropout)).to(torch.bfloat16)
    else:
        attn_weights_dropped = attn_weights

    # Value states
    value_states = torch.randn(
        batch_size, num_key_value_heads, seq_len_kv, head_dim,
        dtype=torch.bfloat16, device=device
    )

    return {
        "grad_attn_output": grad_attn_output,
        "attn_weights": attn_weights,
        "attn_weights_dropped": attn_weights_dropped,
        "value_states": value_states,
        "dropout_mask": dropout_mask,
        "attention_dropout": attention_dropout,
    }


@torch.no_grad()
def run(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    """Backward pass for attention softmax, dropout, and value matmul.

    Computes gradients through:
    1. Transpose gradient
    2. Batched matmul gradients
    3. Dropout gradient
    4. Softmax gradient
    5. GQA gradient aggregation
    """
    num_attention_heads = 80
    num_key_value_heads = 8
    num_key_value_groups = num_attention_heads // num_key_value_heads

    batch_size = grad_attn_output.shape[0]
    seq_len_q = grad_attn_output.shape[1]
    seq_len_kv = value_states.shape[2]
    head_dim = value_states.shape[3]

    # Expand value states for GQA
    value_states_expanded = value_states
    if num_key_value_groups > 1:
        value_states_expanded = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, head_dim
        ).reshape(batch_size, num_attention_heads, seq_len_kv, head_dim)

    # 1. Transpose gradient: (batch, seq_q, heads, head_dim) -> (batch, heads, seq_q, head_dim)
    grad_attn_output_transposed = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. Gradient w.r.t. attn_weights_dropped from matmul
    # Forward: attn_output = attn_weights_dropped @ value_states_expanded
    # grad_attn_weights_dropped = grad_attn_output @ value_states_expanded^T
    grad_attn_weights_dropped = torch.matmul(
        grad_attn_output_transposed,
        value_states_expanded.to(torch.float32).transpose(-2, -1)
    )

    # 3. Gradient through dropout
    # Forward: attn_weights_dropped = attn_weights * mask / (1 - p)
    # Backward: grad_attn_weights = grad_attn_weights_dropped * mask / (1 - p)
    if attention_dropout > 0.0:
        grad_attn_weights = grad_attn_weights_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        grad_attn_weights = grad_attn_weights_dropped

    # 4. Gradient through softmax
    # Using stable formulation: grad_input = softmax * (grad_output - sum(grad_output * softmax))
    attn_weights_f32 = attn_weights.to(torch.float32)
    sum_term = (grad_attn_weights * attn_weights_f32).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights_f32 * (grad_attn_weights - sum_term)
    grad_attn_scores = grad_attn_scores.to(torch.bfloat16)

    # 5. Gradient w.r.t. value_states_expanded from matmul
    # grad_value_states_expanded = attn_weights_dropped^T @ grad_attn_output
    grad_value_states_expanded = torch.matmul(
        attn_weights_dropped.to(torch.float32).transpose(-2, -1),
        grad_attn_output_transposed
    )

    # 6. GQA gradient aggregation
    if num_key_value_groups > 1:
        grad_value_states_expanded = grad_value_states_expanded.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, head_dim
        )
        grad_value_states = grad_value_states_expanded.sum(dim=2)
    else:
        grad_value_states = grad_value_states_expanded

    grad_value_states = grad_value_states.to(torch.bfloat16)

    return grad_attn_scores, grad_value_states
