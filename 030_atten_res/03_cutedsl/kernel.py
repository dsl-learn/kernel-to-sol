import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# ---------------------------------------------------------------------------
# CuTe DSL elementwise-add kernel (projected + residual → output)
#
# Design:
#  - GEMM (attn_output @ o_proj_weight.T) is handled by torch.matmul (cuBLAS)
#  - The residual add uses a custom CuTe DSL kernel compiled to cubin via
#    cute.compile(), which avoids a second global-memory round-trip for the
#    projected tensor.
#  - 128-bit vectorised copy (8 × bfloat16) per thread, (4×32) thread layout
#    → each tile covers (16, 256) elements with 128 threads.
#  - mark_layout_dynamic() lets the same cubin handle all (M, N) shapes.
# ---------------------------------------------------------------------------

_compiled_add = None   # cached cubin, initialised on first call


@cute.kernel
def _add_kernel(gA, gB, gC, cC, shape, thr_layout, val_layout):
    """Per-CTA kernel: C[tile] = A[tile] + B[tile] with boundary predication."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk_coord = ((None, None), bidx)

    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]

    copy_atom   = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    tiled_cpy_A = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_cpy_B = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    tiled_cpy_C = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    thr_A = tiled_cpy_A.get_slice(tidx)
    thr_B = tiled_cpy_B.get_slice(tidx)
    thr_C = tiled_cpy_C.get_slice(tidx)

    thrA = thr_A.partition_S(blkA)
    thrB = thr_B.partition_S(blkB)
    thrC = thr_C.partition_S(blkC)

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    # Boundary predication
    thrCrd  = thr_C.partition_S(blkCrd)
    frgPred = cute.make_rmem_tensor(thrCrd.shape, cutlass.Boolean)
    for i in range(0, cute.size(frgPred), 1):
        frgPred[i] = cute.elem_less(thrCrd[i], shape)

    cute.copy(copy_atom, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom, thrB, frgB, pred=frgPred)
    frgC.store(frgA.load() + frgB.load())
    cute.copy(copy_atom, frgC, thrC, pred=frgPred)


@cute.jit
def _elementwise_add_2d(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    """
    JIT-compiled 2-D elementwise add, compiled to cubin via cute.compile().

    Thread / value layout:
      thr_layout  = (4, 32)   → 128 threads per CTA
      val_layout  = (4, 8)    → 8 bfloat16 elements per thread (128-bit vector)
      tile (M, N) = (16, 256) → 4096 elements per CTA per iteration
    """
    dtype       = mA.element_type
    vector_size = copy_bits // dtype.width          # 8 for bfloat16

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)
    cC = cute.zipped_divide(cute.make_identity_tensor(mC.shape), tiler=tiler_mn)

    _add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


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
    Fused CuTe DSL implementation: output = attn_output @ o_proj_weight.T + residual

    - GEMM     : torch.matmul (cuBLAS), no extra kernel launch
    - Residual : CuTe DSL vectorised add, compiled once to cubin
    """
    global _compiled_add

    original_shape = attn_output.shape
    M = attn_output.numel() // attn_output.shape[-1]
    K = attn_output.shape[-1]
    N = o_proj_weight.shape[0]

    # GEMM via cuBLAS
    projected = torch.matmul(
        attn_output.contiguous().view(M, K),
        o_proj_weight.t(),
    )                                               # (M, N) bfloat16

    # Residual add via CuTe DSL cubin
    res_flat = residual.contiguous().view(M, N)
    output   = torch.empty_like(projected)

    proj_c = from_dlpack(projected).mark_layout_dynamic()
    res_c  = from_dlpack(res_flat).mark_layout_dynamic()
    out_c  = from_dlpack(output).mark_layout_dynamic()

    if _compiled_add is None:
        _compiled_add = cute.compile(_elementwise_add_2d, proj_c, res_c, out_c)

    _compiled_add(proj_c, res_c, out_c)

    return output.view(original_shape)
