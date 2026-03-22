# Fused attention output projection + residual add, implemented with CuTeDSL.
#
# Operation:  C = A @ W^T + R
#   A : (M, K) bfloat16   — attn_output, M = batch * seq_len
#   W : (N, K) bfloat16   — o_proj_weight (kernel treats it as B^T internally)
#   R : (M, N) bfloat16   — residual
#   C : (M, N) bfloat16   — output
#
# Fixed shapes: M = 16*512 = 8192,  N = K = 2560
#
# Layout design rationale
# -----------------------
#  CTA tiler  (bM=128, bN=128, bK=32):
#    - M=8192 / 128 = 64 CTAs,  N=2560 / 128 = 20 CTAs  → 1280 CTAs total
#    - K=2560 / 32  = 80 K-loop iterations per CTA
#  Shared memory (3-stage pipeline):
#    - sA (128, 32, 3)  M-major + 4-element bank-conflict padding
#      → (128+4)*32*3 * 2 B = 25 344 B ≈ 25 KB
#    - sB (128, 32, 3)  N-major, no padding needed
#      → 128*32*3 * 2 B = 24 576 B ≈ 24 KB
#    - Total ≈ 49 KB — well within H100 (228 KB) and B200 (256 KB) limits
#  Thread copy layout:
#    - num_threads=256, bK=32 → tA/tB = (8, 32): 256 threads cover a (8,32) chunk
#    - 128 / 8 = 16 copy steps per thread per stage → moderate register pressure
#  MMA:
#    - MmaUniversalOp(Float32): scalar FMA with fp32 accumulation
#    - Works for bfloat16 inputs; bf16 tensor-core MMA is a drop-in replacement
#  Residual fusion:
#    - After the K-loop, load R for this CTA tile via the same MMA C-partitioning
#    - Add to fp32 accumulator before the bfloat16 store
#    - Eliminates a separate elementwise kernel pass

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

import torch
from typing import Tuple


# ---------------------------------------------------------------------------
# Kernel class
# ---------------------------------------------------------------------------

class FusedOProjResidual:
    """CuTeDSL kernel: C = A @ W^T + R  (all bfloat16, fp32 accumulation)."""

    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 32),
        num_stages: int = 3,
        num_threads: int = 256,
    ):
        self._cta_tiler = cta_tiler
        self._num_stages = num_stages
        self._num_threads = num_threads
        self._bM, self._bN, self._bK = cta_tiler
        assert num_threads % 16 == 0
        assert self._bM % 16 == 0
        assert self._bN % 16 == 0
        assert num_stages >= 3

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,   # (M, K) bfloat16
        mW: cute.Tensor,   # (N, K) bfloat16  — kernel uses it as B = W^T
        mR: cute.Tensor,   # (M, N) bfloat16
        mC: cute.Tensor,   # (M, N) bfloat16  — output
    ):
        # ---------------------------------------------------------------
        # Shared memory layouts
        # sA: M-major with 4-element padding (A is K-major in gmem →
        #     bank conflicts without padding when loading to M-major smem)
        # sB: N-major, no padding (B is already N-major in smem)
        # ---------------------------------------------------------------
        padding_a = 4
        padding_b = 0

        sA_layout = cute.make_layout(
            (self._bM, self._bK, self._num_stages),
            stride=(1, self._bM + padding_a,
                    self._bK * (self._bM + padding_a)),
        )
        sB_layout = cute.make_layout(
            (self._bN, self._bK, self._num_stages),
            stride=(1, self._bN + padding_b,
                    self._bK * (self._bN + padding_b)),
        )

        # ---------------------------------------------------------------
        # Async copy tile layouts and atoms
        # tA/tB: each thread handles one element (non-vectorised cp.async)
        # ---------------------------------------------------------------
        tA = cute.make_layout(
            (self._num_threads // self._bK, self._bK),
            stride=(self._bK, 1),
        )
        tB = cute.make_layout(
            (self._num_threads // self._bK, self._bK),
            stride=(self._bK, 1),
        )
        vA = cute.make_layout((1, 1))
        vB = cute.make_layout((1, 1))

        atom_cp_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width,
        )
        atom_cp_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mW.element_type,
            num_bits_per_copy=mW.element_type.width,
        )
        tiled_copy_A = cute.make_tiled_copy_tv(atom_cp_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_cp_B, tB, vB)

        # ---------------------------------------------------------------
        # MMA: scalar FMA with fp32 accumulator
        #   atoms_layout (T//16, 16, 1): T threads, 16-wide warp group
        #   permutation_tiler: each thread owns 4 consecutive elements
        # ---------------------------------------------------------------
        atoms_layout = cute.make_layout(
            (self._num_threads // 16, 16, 1),
            stride=(16, 1, 0),
        )
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        perm_M = cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1))
        perm_N = cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1))
        tiled_mma = cute.make_tiled_mma(
            op, atoms_layout, permutation_mnk=(perm_M, perm_N, None)
        )

        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        self.kernel(
            mA, mW, mR, mC,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=grid_dim,
            block=(cute.size(atoms_layout), 1, 1),
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mW: cute.Tensor,
        mR: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)
        thr_mma = tiled_mma.get_slice(tidx)

        # ------------------------------------------------------------------
        # Global tiles for this CTA
        #   gA: (bM, bK, k)   gW: (bN, bK, k)
        #   gC: (bM, bN)      gR: (bM, bN)
        # ------------------------------------------------------------------
        gA = cute.local_tile(
            mA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        gW = cute.local_tile(
            mW, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        gC = cute.local_tile(
            mC, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )
        gR = cute.local_tile(
            mR, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )

        # Move pointers so the first (irregular) K-tile is processed first
        residue_k = mA.shape[1] - cutlass.Int32(self._bK) * gA.shape[2]
        gA = cute.domain_offset((0, residue_k, 0), gA)
        gW = cute.domain_offset((0, residue_k, 0), gW)

        # ------------------------------------------------------------------
        # Shared memory buffers
        # ------------------------------------------------------------------
        smem = utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mW.element_type, sB_layout, 16)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gW)
        tBsB = thr_copy_B.partition_D(sB)

        # ------------------------------------------------------------------
        # Predicate tensors for boundary checking
        # ------------------------------------------------------------------
        mcA = cute.make_identity_tensor(mA.shape)
        mcB = cute.make_identity_tensor(mW.shape)
        cA = cute.local_tile(
            mcA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        cB = cute.local_tile(
            mcB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        cA = cute.domain_offset((0, residue_k, 0), cA)
        cB = cute.domain_offset((0, residue_k, 0), cB)
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)

        # Predicate: m/n bounds only (k=0 slice, k-stride=0)
        tApA = cute.make_fragment(
            cute.make_layout(
                (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_fragment(
            cute.make_layout(
                (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        # Predicate: m/n/k bounds for the residue k-tile
        tApA_residue_k = cute.make_fragment(
            cute.make_layout(
                (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
                stride=(
                    cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                    cute.size(tAsA, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        tBpB_residue_k = cute.make_fragment(
            cute.make_layout(
                (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                stride=(
                    cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                    cute.size(tBsB, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )

        for rv in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rv, m, 0] = cute.elem_less(
                    tAcA[(0, rv), m, 0, 0][0], mA.shape[0]
                )
        for rv in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rv, n, 0] = cute.elem_less(
                    tBcB[(0, rv), n, 0, 0][0], mW.shape[0]
                )
        for rv in range(tApA_residue_k.shape[0]):
            for m in range(tApA_residue_k.shape[1]):
                for k in range(tApA_residue_k.shape[2]):
                    coord = tAcA[(0, rv), m, k, 0]
                    tApA_residue_k[rv, m, k] = cute.elem_less(
                        (coord[0], cutlass.Int32(-1)), (mA.shape[0], coord[1])
                    )
        for rv in range(tBpB_residue_k.shape[0]):
            for n in range(tBpB_residue_k.shape[1]):
                for k in range(tBpB_residue_k.shape[2]):
                    coord = tBcB[(0, rv), n, k, 0]
                    tBpB_residue_k[rv, n, k] = cute.elem_less(
                        (coord[0], cutlass.Int32(-1)), (mW.shape[0], coord[1])
                    )

        # ------------------------------------------------------------------
        # Prologue: prefetch (num_stages - 1) K-tiles into smem
        # ------------------------------------------------------------------
        k_pipe_max = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)

        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA_residue_k,
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, 0],
            pred=tBpB_residue_k,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < k_tile_count
            else cutlass.Int32(0)
        )

        for k_tile in range(1, k_pipe_max - 1):
            if k_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < k_tile_count
                else cutlass.Int32(0)
            )
            cute.arch.cp_async_commit_group()

        if k_tile_count < k_pipe_max:
            for rv in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rv, m, 0] = cutlass.Boolean(0)
            for rv in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rv, n, 0] = cutlass.Boolean(0)

        # ------------------------------------------------------------------
        # MMA register partitioning and accumulator init
        # ------------------------------------------------------------------
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(k_pipe_max - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]

        # ------------------------------------------------------------------
        # Register prefetch
        # ------------------------------------------------------------------
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(k_pipe_max - 2)
            cute.arch.barrier()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        # ------------------------------------------------------------------
        # Mainloop
        # ------------------------------------------------------------------
        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(k_pipe_max - 2)
                    cute.arch.barrier()

                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(tCsA_p[None, None, k_block_next], tCrA[None, None, k_block_next])
                cute.autovec_copy(tCsB_p[None, None, k_block_next], tCrB[None, None, k_block_next])

                if k_block == 0:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )

                cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                if k_block == 0:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, gmem_pipe_read],
                        tBsB[None, None, None, smem_pipe_write],
                        pred=tBpB,
                    )
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(1)
                    )

        # ------------------------------------------------------------------
        # Epilogue: fused residual add + bfloat16 store
        # ------------------------------------------------------------------
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Load residual tile for this CTA using the same C-partitioning
        tCgR = thr_mma.partition_C(gR)

        # Add residual (bfloat16) to fp32 accumulator in-place
        for i in range(cute.size(tCrC)):
            tCrC[i] = tCrC[i] + tCgR[i].to(cutlass.Float32)

        # Predicate for output bounds (handles tail M / N)
        cC = cute.make_identity_tensor(gC.shape)
        tCpC = thr_mma.partition_C(cC)
        predC = cute.make_fragment(tCrC.layout, cutlass.Boolean)
        residue_m = mC.shape[0] - cutlass.Int32(self._bM) * bidx
        residue_n = mC.shape[1] - cutlass.Int32(self._bN) * bidy
        for i in range(cute.size(tCrC.shape)):
            predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))

        # Copy from fp32 registers to bfloat16 global memory
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC, pred=predC)


# ---------------------------------------------------------------------------
# Host entry point
# ---------------------------------------------------------------------------

_fused_op = FusedOProjResidual(cta_tiler=(128, 128, 32), num_stages=3, num_threads=256)


@cute.jit
def _launch(
    mA: cute.Tensor,
    mW: cute.Tensor,
    mR: cute.Tensor,
    mC: cute.Tensor,
):
    _fused_op(mA, mW, mR, mC)


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Fused CuTeDSL implementation of attention output projection + residual add.

    Performs: output = attn_output @ o_proj_weight.T + residual

    All tensors must be bfloat16, contiguous, on CUDA.
    """
    B, S, H_in = attn_output.shape
    H_out = o_proj_weight.shape[0]
    M, N, K = B * S, H_out, H_in

    A = attn_output.reshape(M, K).contiguous()   # (M, K)
    W = o_proj_weight.contiguous()               # (N, K)  — kernel transposes via layout
    R = residual.reshape(M, N).contiguous()      # (M, N)
    C = torch.empty(M, N, dtype=torch.bfloat16, device=attn_output.device)

    _launch(A, W, R, C)

    return C.view(B, S, N)
