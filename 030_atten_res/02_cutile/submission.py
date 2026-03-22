import math
import logging
from typing import Any

import torch
import cuda.tile as ct
from cuda.tile._cext import default_tile_context
from cuda.tile._exception import TileCompilerExecutionError, TileCompilerTimeoutError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Autotuner (adapted from cutile-learn/tutorials/autotuner.py)
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, *, num_ctas=None, occupancy=None, opt_level=3, **kwargs):
        self.kwargs = dict(kwargs)
        self.num_ctas = num_ctas
        self.occupancy = occupancy
        self.opt_level = opt_level

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(name)

    def __str__(self):
        parts = [f"{k}={v}" for k, v in self.kwargs.items()]
        parts += [f"num_ctas={self.num_ctas}", f"occupancy={self.occupancy}"]
        return f"Config({', '.join(parts)})"


def _time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
    stream.synchronize()
    for _ in range(warmup):
        run_once(get_args())
    args_per_run = [get_args() for _ in range(rep)]
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for args in args_per_run:
        run_once(args)
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) / max(1, rep)


class Autotuner:
    def __init__(self, configs: list[Config]):
        self._configs = configs
        self._cache: dict = {}

    def __call__(self, stream, grid_fn, kernel, args_fn) -> Any:
        # Build a cache key from the first args' shapes/dtypes
        sample_args = args_fn(self._configs[0])
        key = tuple(
            (a.shape, str(a.dtype)) if hasattr(a, "shape") else a
            for a in sample_args
        )

        if key in self._cache:
            best_cfg, best_grid, best_kernel = self._cache[key]
        else:
            best_time, best_cfg, best_grid, best_kernel = float("inf"), None, None, None
            for cfg in self._configs:
                args = args_fn(cfg)
                grid = grid_fn(cfg)
                updated_kernel = ct.kernel(
                    kernel._pyfunc,
                    num_ctas=cfg.num_ctas,
                    occupancy=cfg.occupancy,
                    opt_level=cfg.opt_level,
                )
                try:
                    old_timeout = default_tile_context.config.compiler_timeout_sec
                    default_tile_context.config.compiler_timeout_sec = 30
                    t = _time_ms(
                        lambda a: ct.launch(stream, grid, updated_kernel, a),
                        get_args=lambda cfg=cfg: args_fn(cfg),
                        stream=stream,
                    )
                    default_tile_context.config.compiler_timeout_sec = old_timeout
                except (TileCompilerTimeoutError, TileCompilerExecutionError) as e:
                    logger.debug(f"Config {cfg} skipped: {e}")
                    continue
                if t < best_time:
                    best_time, best_cfg = t, cfg
                    best_grid, best_kernel = grid, updated_kernel
                    logger.debug(f"New best {cfg}: {t:.3f} ms")

            if best_cfg is None:
                raise RuntimeError("No valid cuTile config found")
            self._cache[key] = (best_cfg, best_grid, best_kernel)

        ct.launch(stream, best_grid, best_kernel, args_fn(best_cfg))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

ConstInt = ct.Constant[int]


def _swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M):
    """L2-cache-friendly 2-D CTA swizzle (same pattern as cuTile matmul tutorial)."""
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_M)
    num_bid_n = ct.cdiv(N, TILE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def _fused_o_proj_residual_kernel(
    A, B, R, C,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
):
    """
    Fused attention output projection + residual add.

    Computes: C = A @ B + R

    A : (M, K)  — attn_output flattened, M = batch * seq_len, K = hidden_size
    B : (K, N)  — o_proj_weight.T, contiguous
    R : (M, N)  — residual flattened
    C : (M, N)  — output

    Each CTA owns a (TILE_M, TILE_N) output tile.  The K-loop accumulates in
    float32; the residual is fused before the final bfloat16 store.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    PAD = ct.PaddingMode.ZERO

    bid_m, bid_n = _swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_M, TILE_K))
    acc = ct.full((TILE_M, TILE_N), 0, dtype=ct.float32)

    # bfloat16 -> use native dtype for mma (no tf32 conversion needed)
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bid_m, k), shape=(TILE_M, TILE_K), padding_mode=PAD)
        b = ct.load(B, index=(k, bid_n), shape=(TILE_K, TILE_N), padding_mode=PAD)
        acc = ct.mma(a, b, acc)

    # Fuse residual add before store — eliminates a separate elementwise pass
    r = ct.load(R, index=(bid_m, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD)
    out = acc + r.astype(ct.float32)

    ct.store(C, index=(bid_m, bid_n), tile=ct.astype(out, C.dtype))


# ---------------------------------------------------------------------------
# Config search space
# ---------------------------------------------------------------------------

def _configs():
    cap = torch.cuda.get_device_capability()
    if cap in [(12, 0), (12, 1)]:
        # sm_120 / sm_121  (RTX 5090)
        return [
            Config(TILE_M=128, TILE_N=64,  TILE_K=64, num_ctas=1, occupancy=1),
            Config(TILE_M=128, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=1),
            Config(TILE_M=64,  TILE_N=128, TILE_K=64, num_ctas=1, occupancy=1),
            Config(TILE_M=128, TILE_N=64,  TILE_K=32, num_ctas=1, occupancy=2),
            Config(TILE_M=128, TILE_N=128, TILE_K=32, num_ctas=1, occupancy=2),
        ]
    else:
        # sm_100  (Blackwell B200)
        return [
            Config(TILE_M=128, TILE_N=128, TILE_K=32, num_ctas=1, occupancy=1),
            Config(TILE_M=256, TILE_N=256, TILE_K=64, num_ctas=2, occupancy=1),
            Config(TILE_M=256, TILE_N=256, TILE_K=64, num_ctas=4, occupancy=1),
            Config(TILE_M=512, TILE_N=256, TILE_K=64, num_ctas=2, occupancy=1),
            Config(TILE_M=256, TILE_N=128, TILE_K=64, num_ctas=2, occupancy=1),
        ]


_autotuner = None  # lazily initialised


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
    Fused cuTile implementation of attention output projection + residual add.

    Performs: output = attn_output @ o_proj_weight.T + residual

    All tensors must be bfloat16, contiguous, on CUDA.
    """
    global _autotuner
    if _autotuner is None:
        _autotuner = Autotuner(_configs())

    B, S, H_in = attn_output.shape
    H_out = o_proj_weight.shape[0]

    M = B * S
    N = H_out
    K = H_in

    # Flatten and ensure contiguous
    A = attn_output.reshape(M, K).contiguous()         # (M, K)
    Wt = o_proj_weight.T.contiguous()                  # (K, N)  — transposed weight
    R = residual.reshape(M, N).contiguous()             # (M, N)
    C = torch.empty(M, N, dtype=torch.bfloat16, device=attn_output.device)

    stream = torch.cuda.current_stream()

    _autotuner(
        stream,
        grid_fn=lambda cfg: (
            math.ceil(M / cfg.TILE_M) * math.ceil(N / cfg.TILE_N),
            1, 1,
        ),
        kernel=_fused_o_proj_residual_kernel,
        args_fn=lambda cfg: (A, Wt, R, C, cfg.TILE_M, cfg.TILE_N, cfg.TILE_K),
    )

    return C.view(B, S, N)
