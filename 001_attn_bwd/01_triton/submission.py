import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: fused dScore  (dO @ V^T  +  dropout backward  +  softmax backward)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ---- small tiles (T4 / low-register-pressure baseline) ----
        triton.Config({"BLOCK_Q": 16, "BLOCK_KV":  64}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 128}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV":  64}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=8,  num_stages=2),
        # ---- medium tiles (good tensor-core utilisation) ----
        triton.Config({"BLOCK_Q": 16, "BLOCK_KV":  64}, num_warps=4,  num_stages=3),
        triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 128}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV":  64}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV":  64}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8,  num_stages=3),
        # ---- large tiles (B200: 228 KB SRAM, deep pipeline) ----
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 256}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 256}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=16, num_stages=5),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 256}, num_warps=16, num_stages=5),
    ],
    key=["Sq", "Skv"],
)
@triton.jit
def _fused_grad_scores_kernel(
    # Pointers
    dO_ptr, V_ptr, P_ptr, mask_ptr, dS_ptr,
    # dO strides: [B, Sq, H, D]
    s_dO_b, s_dO_sq, s_dO_h, s_dO_d,
    # V strides: [B, Hkv, Skv, D]
    s_V_b, s_V_hkv, s_V_skv, s_V_d,
    # P strides: [B, H, Sq, Skv]
    s_P_b, s_P_h, s_P_sq, s_P_skv,
    # mask strides: [B, H, Sq, Skv]
    s_m_b, s_m_h, s_m_sq, s_m_skv,
    # dS strides: [B, H, Sq, Skv]
    s_dS_b, s_dS_h, s_dS_sq, s_dS_skv,
    # Sizes
    H, Sq, Skv,
    num_groups: tl.constexpr,
    inv_keep_prob,
    has_dropout: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    For each (b, h, q_tile), computes grad_attn_scores in two passes over KV:

    Pass 1:  dP[q,kv] = dO[q,:] @ V[kv,:]^T
             if dropout: dP *= mask * inv_keep_prob
             sum_term[q] += sum_kv( dP[q,kv] * P[q,kv] )

    Pass 2:  dS[q,kv] = P[q,kv] * ( dP[q,kv] - sum_term[q] )

    Grid: (B*H, ceil(Sq/BLOCK_Q))
    """
    BLOCK_D: tl.constexpr = 128  # head_dim always 128

    bh = tl.program_id(0)
    q_tile = tl.program_id(1)

    b = bh // H
    h = bh % H
    hkv = h // num_groups

    q_start = q_tile * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < Sq
    d_offs = tl.arange(0, BLOCK_D)

    # Load dO tile once: [BLOCK_Q, BLOCK_D] as bfloat16
    dO_tile = tl.load(
        dO_ptr + b * s_dO_b + q_offs[:, None] * s_dO_sq
        + h * s_dO_h + d_offs[None, :] * s_dO_d,
        mask=q_mask[:, None],
        other=0.0,
    )  # bfloat16

    # ---- Pass 1: accumulate sum_term ----------------------------------------
    sum_term = tl.zeros([BLOCK_Q], dtype=tl.float32)

    for kv_start in tl.range(0, Skv, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < Skv

        # Load V[b, hkv, kv, d]: [BLOCK_KV, BLOCK_D] as bfloat16
        V_tile = tl.load(
            V_ptr + b * s_V_b + hkv * s_V_hkv
            + kv_offs[:, None] * s_V_skv + d_offs[None, :] * s_V_d,
            mask=kv_mask[:, None],
            other=0.0,
        )  # bfloat16, [BLOCK_KV, BLOCK_D]

        # dP_drop = dO @ V^T : [BLOCK_Q, BLOCK_KV], accumulates in float32
        dP_drop = tl.dot(dO_tile, tl.trans(V_tile))  # float32

        if has_dropout:
            m_tile = tl.load(
                mask_ptr + b * s_m_b + h * s_m_h
                + q_offs[:, None] * s_m_sq + kv_offs[None, :] * s_m_skv,
                mask=q_mask[:, None] & kv_mask[None, :],
                other=0,
            ).to(tl.float32)
            dP = dP_drop * m_tile * inv_keep_prob
        else:
            dP = dP_drop

        P_tile = tl.load(
            P_ptr + b * s_P_b + h * s_P_h
            + q_offs[:, None] * s_P_sq + kv_offs[None, :] * s_P_skv,
            mask=q_mask[:, None] & kv_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # row-sum: [BLOCK_Q, BLOCK_KV] -> [BLOCK_Q]
        sum_term += tl.sum(dP * P_tile, axis=1)

    # ---- Pass 2: compute dS and store ---------------------------------------
    for kv_start in tl.range(0, Skv, BLOCK_KV):
        kv_offs = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < Skv

        V_tile = tl.load(
            V_ptr + b * s_V_b + hkv * s_V_hkv
            + kv_offs[:, None] * s_V_skv + d_offs[None, :] * s_V_d,
            mask=kv_mask[:, None],
            other=0.0,
        )

        dP_drop = tl.dot(dO_tile, tl.trans(V_tile))

        if has_dropout:
            m_tile = tl.load(
                mask_ptr + b * s_m_b + h * s_m_h
                + q_offs[:, None] * s_m_sq + kv_offs[None, :] * s_m_skv,
                mask=q_mask[:, None] & kv_mask[None, :],
                other=0,
            ).to(tl.float32)
            dP = dP_drop * m_tile * inv_keep_prob
        else:
            dP = dP_drop

        P_tile = tl.load(
            P_ptr + b * s_P_b + h * s_P_h
            + q_offs[:, None] * s_P_sq + kv_offs[None, :] * s_P_skv,
            mask=q_mask[:, None] & kv_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        dS = P_tile * (dP - sum_term[:, None])

        tl.store(
            dS_ptr + b * s_dS_b + h * s_dS_h
            + q_offs[:, None] * s_dS_sq + kv_offs[None, :] * s_dS_skv,
            dS.to(tl.bfloat16),
            mask=q_mask[:, None] & kv_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Kernel 2: fused dV  (attn_weights_dropped^T @ dO  +  GQA group sum)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ---- small tiles ----
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64},  num_warps=8, num_stages=2),
        # ---- medium tiles ----
        triton.Config({"BLOCK_Q":  32, "BLOCK_KV":  64}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_Q":  64, "BLOCK_KV":  64}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_Q":  64, "BLOCK_KV": 128}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV":  32}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV":  64}, num_warps=8,  num_stages=3),
        # ---- large tiles (B200) ----
        triton.Config({"BLOCK_Q":  64, "BLOCK_KV":  64}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_Q":  64, "BLOCK_KV": 128}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV":  64}, num_warps=8,  num_stages=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV":  64}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 128}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV":  64}, num_warps=16, num_stages=5),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 128}, num_warps=16, num_stages=5),
    ],
    key=["Sq", "Skv"],
)
@triton.jit
def _fused_grad_value_kernel(
    # Pointers
    dO_ptr, Pd_ptr, dV_ptr,
    # dO strides: [B, Sq, H, D]
    s_dO_b, s_dO_sq, s_dO_h, s_dO_d,
    # P_drop strides: [B, H, Sq, Skv]
    s_Pd_b, s_Pd_h, s_Pd_sq, s_Pd_skv,
    # dV strides: [B, Hkv, Skv, D]
    s_dV_b, s_dV_hkv, s_dV_skv, s_dV_d,
    # Sizes
    Hkv, Sq, Skv,
    num_groups: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Computes dV[b, hkv, kv, d] = sum_{g} sum_q  P_drop[b, h, q, kv] * dO[b, q, h, d]
    where h = hkv * num_groups + g.

    Uses GEMM: for each (b, hkv, kv_tile), accumulates over all (groups, q_tiles).

    Grid: (B * Hkv, ceil(Skv / BLOCK_KV))
    """
    BLOCK_D: tl.constexpr = 128  # head_dim always 128

    b_hkv = tl.program_id(0)
    kv_tile = tl.program_id(1)

    b = b_hkv // Hkv
    hkv = b_hkv % Hkv

    kv_start = kv_tile * BLOCK_KV
    kv_offs = kv_start + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offs < Skv
    d_offs = tl.arange(0, BLOCK_D)

    dV_acc = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

    for g in tl.range(0, num_groups):
        h = hkv * num_groups + g

        for q_start in tl.range(0, Sq, BLOCK_Q):
            q_offs = q_start + tl.arange(0, BLOCK_Q)
            q_mask = q_offs < Sq

            # Load P_drop[b, h, q, kv]: [BLOCK_Q, BLOCK_KV] as bfloat16
            Pd_tile = tl.load(
                Pd_ptr + b * s_Pd_b + h * s_Pd_h
                + q_offs[:, None] * s_Pd_sq + kv_offs[None, :] * s_Pd_skv,
                mask=q_mask[:, None] & kv_mask[None, :],
                other=0.0,
            )  # bfloat16

            # Load dO[b, q, h, d]: [BLOCK_Q, BLOCK_D] as bfloat16
            dO_tile = tl.load(
                dO_ptr + b * s_dO_b + q_offs[:, None] * s_dO_sq
                + h * s_dO_h + d_offs[None, :] * s_dO_d,
                mask=q_mask[:, None],
                other=0.0,
            )  # bfloat16

            # dV_acc += P_drop^T @ dO : [BLOCK_KV, BLOCK_Q] @ [BLOCK_Q, BLOCK_D]
            dV_acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    tl.store(
        dV_ptr + b * s_dV_b + hkv * s_dV_hkv
        + kv_offs[:, None] * s_dV_skv + d_offs[None, :] * s_dV_d,
        dV_acc.to(tl.bfloat16),
        mask=kv_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    """Fused Triton backward pass for attention softmax + dropout + value matmul."""
    B, Sq, H, D = grad_attn_output.shape
    Hkv = value_states.shape[1]
    Skv = value_states.shape[2]
    num_groups = H // Hkv

    has_dropout = attention_dropout > 0.0
    inv_keep_prob = float(1.0 / (1.0 - attention_dropout)) if has_dropout else 1.0

    # Allocate outputs
    grad_attn_scores = torch.empty(
        B, H, Sq, Skv, dtype=torch.bfloat16, device=grad_attn_output.device
    )
    grad_value_states = torch.empty(
        B, Hkv, Skv, D, dtype=torch.bfloat16, device=grad_attn_output.device
    )

    # Kernel 1: grad_attn_scores
    grid1 = lambda meta: (B * H, triton.cdiv(Sq, meta["BLOCK_Q"]))
    _fused_grad_scores_kernel[grid1](
        grad_attn_output, value_states, attn_weights, dropout_mask, grad_attn_scores,
        *grad_attn_output.stride(),   # s_dO_b, s_dO_sq, s_dO_h, s_dO_d
        *value_states.stride(),       # s_V_b, s_V_hkv, s_V_skv, s_V_d
        *attn_weights.stride(),       # s_P_b, s_P_h, s_P_sq, s_P_skv
        *dropout_mask.stride(),       # s_m_b, s_m_h, s_m_sq, s_m_skv
        *grad_attn_scores.stride(),   # s_dS_b, s_dS_h, s_dS_sq, s_dS_skv
        H=H, Sq=Sq, Skv=Skv,
        num_groups=num_groups,
        inv_keep_prob=inv_keep_prob,
        has_dropout=has_dropout,
    )

    # Kernel 2: grad_value_states
    grid2 = lambda meta: (B * Hkv, triton.cdiv(Skv, meta["BLOCK_KV"]))
    _fused_grad_value_kernel[grid2](
        grad_attn_output, attn_weights_dropped, grad_value_states,
        *grad_attn_output.stride(),       # s_dO_b, s_dO_sq, s_dO_h, s_dO_d
        *attn_weights_dropped.stride(),   # s_Pd_b, s_Pd_h, s_Pd_sq, s_Pd_skv
        *grad_value_states.stride(),      # s_dV_b, s_dV_hkv, s_dV_skv, s_dV_d
        Hkv=Hkv, Sq=Sq, Skv=Skv,
        num_groups=num_groups,
    )

    return grad_attn_scores, grad_value_states
