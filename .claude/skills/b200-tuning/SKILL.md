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

## GroupNorm / reduction kernel tuning (learned from 002_vae_conv2d)

### num_warps and num_stages

- Default `num_warps=8` leaves half of B200's warp groups idle → always use `num_warps=16`
- Always set `num_stages=4` on memory-bound reduction kernels; default of 1 stalls on HBM latency

### Eliminate atomics in spatial-split stats kernels

Original pattern (bad): S programs each write partial sum/sum_sq to the same group slot via `tl.atomic_add` → serialization and contention.

Fix: give each program its **own unique slot** in a `partial[N*G*S, 2]` buffer, then reduce inline in the normalize kernel:

```python
# Stats kernel — no atomic, each pid owns pid*2 slot
tl.store(partial_ptr + pid * 2 + 0, tl.sum(_sum))
tl.store(partial_ptr + pid * 2 + 1, tl.sum(_ssq))

# Norm kernel — O(S) inline reduce before normalizing
total_sum = 0.0
for i in range(S):
    total_sum += tl.load(partial_ptr + (group_id * S + i) * 2 + 0)
```

### Fuse stats + normalize into a single kernel (Path A)

When N*G is large enough to saturate SMs, eliminate the intermediate stats tensor entirely — accumulate sum/sum_sq in register vectors, reduce with `tl.sum`, then normalize in a second loop within the same kernel launch. Saves one global read+write round-trip and one kernel launch.

```python
_sum = tl.zeros([BLOCK], dtype=tl.float32)
_ssq = tl.zeros([BLOCK], dtype=tl.float32)
for i in ...:          # Pass 1: stats in registers
    _sum += x; _ssq += x * x
mean = tl.sum(_sum) / group_size
rstd = tl.rsqrt(tl.sum(_ssq) / group_size - mean * mean + eps)
for i in ...:          # Pass 2: normalize + activation
    ...
```

### Adaptive dispatch for SM saturation

`grid = (N * G,)` starves the GPU when N*G < NUM_SMS (e.g. batch=1 with large spatial).

Rule: if `N*G >= NUM_SMS` use Path A (fused, one program per group); otherwise use Path B (spatial split into S chunks, no atomics):

```python
NUM_SMS         = 160   # B200
TARGET_PROGRAMS = NUM_SMS * 8   # 1280

if N * G >= NUM_SMS:
    # Path A: fused single-launch, grid = (N*G,)
else:
    S = max(2, TARGET_PROGRAMS // (N * G))
    # Path B: partial stats (no atomic) + inline reduce, grid = (N*G*S,)
```

### Chunk-boundary mask in spatial-split kernels

When `chunk_size < BLOCK`, mask `offs < group_size` alone causes adjacent chunks to overlap → double-counted elements → negative variance → NaN. Always use:

```python
chunk_end = chunk_start + chunk_size
mask = (offs < chunk_end) & (offs < group_size)
```

## Decision tree

- **Low TFLOPS despite large tiles** → check `num_stages`; increase to 4–5
- **Register spills** → reduce tile size or `num_stages`
- **Poor L2 hit rate** → add grid swizzle (`tl.swizzle2d` or manual pid swap)
- **Small batch (M < 512)** → consider persistent kernel to reuse weight tiles
- **Bank conflicts** → pad smem rows by 1 element in arrays fed to `tl.dot`
