---
name: b200-tuning
description: >
  B200 / Blackwell (sm_100a) performance tuning for Triton and CUTLASS kernels.
  Use when the user asks to optimize, benchmark, or tune a kernel for B200,
  mentions TFLOPS targets, tile sizes, pipeline stages, TMA, WGMMA, or asks
  why a kernel is slow on Blackwell.
---

# B200 Performance Tuning

## Hardware reference — NVIDIA B200 (sm_100a)

| Property | Value |
|---|---|
| BF16 compute | ~2.25 PFlops (WGMMA / tcgen05) |
| SRAM per SM | 228 KB |
| L2 cache | 96 MB |
| HBM3e bandwidth | 8 TB/s |
| Warp groups per SM | 4 × 128 threads |
| TMA | Hardware async copy with bounds checking |
| WGMMA atom (BF16) | `64×N×K`, K multiple of 16 |

## Triton

### Starting autotune configs

```python
triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8,  num_stages=4),
triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8,  num_stages=4),
triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=4),
triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=16, num_stages=5),
triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=16, num_stages=5),
```

| Parameter | Guidance |
|---|---|
| `BLOCK_K` | 64 (= 128-byte cache line for bf16) |
| `num_warps` | 16 → 4 warpgroups → full SM occupancy |
| `num_stages` | 4–5: deep pipeline hides HBM latency |
| `key` | Always include problem-size dims, e.g. `key=["M", "N", "K"]` |

### TMA

Use `tl.make_tensor_descriptor` + `tl.load_tensor_descriptor` for **every**
global access. Benefits: hardware bounds checking (no mask overhead), async
prefetch, coalesced access independent of tile alignment.

### Fused epilogue

Do residual add / bias / activation inside the same kernel before the final
store — avoids a separate memory round-trip.

### SMEM budget check

```
(BLOCK_M × BLOCK_K + BLOCK_N × BLOCK_K) × 2 bytes × num_stages ≤ 228 KB
```

Example: (128×64 + 256×64) × 2 × 5 = 122 880 bytes ≈ 120 KB ✓

## CUTLASS / CuTe-DSL

- Architecture flag: `sm_100a`
- Collective: `SM100_64x128x32_F32BF16BF16F32_SS` or larger atom
- Pipeline stages: 4–5 (apply same SRAM budget formula above)
- Epilogue: `LinearCombination` or custom visitor to fuse residual at register
  level (no partial-GEMM global write)

## Decision tree

- **Low TFLOPS despite large tiles** → check `num_stages`; increase to 4–5
- **Register spills** → reduce tile size or `num_stages`
- **Poor L2 hit rate** → add grid swizzle (`tl.swizzle2d` or manual pid swap)
- **Small batch (M < 512)** → consider persistent kernel to reuse weight tiles
- **Bank conflicts** → pad smem rows by 1 element in arrays fed to `tl.dot`
