---
name: kernel-testing
description: >
  Run correctness tests for kernel implementations. Use when the user asks to
  run test.py, validate a kernel, debug a test failure, or check if an
  implementation passes. Covers test commands, failure diagnosis, and Triton
  IR debugging env vars.
disable-model-invocation: true
---

# Kernel Testing

**Never run performance benchmarks unless the user explicitly asks.**

## Run tests

```bash
# Auto-selects best available impl (cutedsl on CUDA, torch_ref otherwise)
python <problem>/test.py

# Test a specific implementation
python <problem>/test.py --impl triton
python <problem>/test.py --impl cutile
python <problem>/test.py --impl cutedsl

# Single shape
python <problem>/test.py --impl triton --batch-size 4 --seq-len 128

# Random stress test
python <problem>/test.py --impl triton --random-tests 20 --seed 42
```

## Diagnose failures

| Symptom | Likely cause |
|---|---|
| `allclose=False`, `max_abs_diff > 0.1` | bf16 accumulation — use float32 in `tl.dot` / GEMM |
| `allclose=False`, `max_abs_diff ≈ 0.01` | Edge tile not masked — check TMA descriptor or explicit mask |
| Wrong output shape | Forgot `.view(B, S, N)` after 2D GEMM |
| CUDA illegal memory access | Input not `.contiguous()` before passing to kernel |
| cubin error: `-arch=compute_` empty | GPU architecture not detected — ignore, env issue |
| Module load error | `*_PATH` constant or `--impl` choice not added to `test.py` |

## Triton IR debugging

| Env var | Effect |
|---|---|
| `TRITON_KERNEL_DUMP=1` | Dump all IR stages to `~/.triton/dump/` |
| `TRITON_KERNEL_DUMP_BEST_CONFIG=1` | Dump only the winning autotuned config |
| `TRITON_PRINT_AUTOTUNING=1` | Readable subdirectory names (combine with DUMP) |
| `TRITON_DUMP_PTXAS_LOG=1` | ptxas register usage and spill report |
| `TRITON_ALWAYS_COMPILE=1` | Bypass cache, force recompilation |
| `TRITON_INTERPRET=1` | Run on CPU (no GPU needed, slow but debuggable) |

```bash
# Debug register spills
TRITON_DUMP_PTXAS_LOG=1 TRITON_ALWAYS_COMPILE=1 python <problem>/test.py --impl triton

# Inspect best autotuned config IR
TRITON_KERNEL_DUMP_BEST_CONFIG=1 TRITON_PRINT_AUTOTUNING=1 python <problem>/test.py --impl triton
```
