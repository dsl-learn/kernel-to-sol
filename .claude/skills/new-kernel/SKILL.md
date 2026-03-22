---
name: new-kernel
description: >
  Implement a new kernel.py for a problem directory. Use when the user asks to
  add a new implementation (triton, cutile, cutedsl, cublaslt) or create a new
  impl directory, or when writing a kernel from scratch given a torch reference.
---

# New Kernel Implementation

## Interface contract (non-negotiable)

Read `<problem>/00_torch_ref/kernel.py` first. Your `kernel.py` must:

- Export one public function `run(...)` with the **exact same signature** as
  the torch reference (same parameter names, order, dtypes).
- Decorate with `@torch.no_grad()`.
- Input tensors are **bfloat16, contiguous, on CUDA**.
- Return a **bfloat16 tensor** matching the reference output shape.
- For multi-output kernels, return in the **same order** as the reference.

## Adding the impl to test.py

Open `<problem>/test.py` and:

1. Add a `<NAME>_PATH` constant alongside the existing `*_PATH` lines.
2. Add the impl name to the `--impl` argparse `choices`.
3. Add an `elif impl_name == "<name>":` branch in the module-loading block.
4. Add `"<name>"` to the `_resolve_impl_name` fallback if it should be the
   CUDA default.

## Correctness checklist

- [ ] Accumulate `tl.dot` in **float32**; cast to bfloat16 only at the final store.
- [ ] For non-power-of-two shapes: use `tl.make_tensor_descriptor` (hardware
  bounds-checking) or explicit masks — never assume aligned sizes.
- [ ] Output shape and dtype match the reference before returning.

## Common pitfalls

| Symptom | Likely cause |
|---|---|
| Large `max_abs_diff` (> 0.1) | Accumulating in bfloat16 instead of float32 |
| Wrong output shape | Forgot to `.view(B, S, N)` after 2D GEMM |
| CUDA error on first run | Input tensor not `.contiguous()` before TMA use |
| Shape mismatch for odd seq_len | Missing boundary mask or TMA bounds |
| Multi-output order wrong | Return order differs from reference |
| NaN output in spatial-split kernel | Chunk-boundary mask bug (see below) |

## Platform submission constraints (sol-execbench)

### Forbidden keyword: `stream`

The submission validator does a **literal keyword scan** of the source file.
The word `stream` anywhere — including comments and docstrings — causes rejection:

```
Value error, Source file 'submission.py' contains the forbidden keyword 'stream'.
CUDA stream usage is not permitted in Python solutions.
```

- Never write `stream` in any comment, docstring, variable name, or string literal.
- Rephrase: "CUDA stream barrier" → "implicit barrier between launches".
- Multiple kernel launches in sequence are **allowed**; only the word is banned.
- **Check before submitting:** `grep -n stream kernel.py`

### Chunk-boundary mask bug in spatial-split kernels

When splitting a group into S chunks and `chunk_size < BLOCK`, the mask
`offs < group_size` alone is **insufficient** — adjacent chunks overlap,
double-count elements, corrupt partial stats, and produce NaN variance.

Always use the tighter mask in both the stats and norm passes:

```python
chunk_end = chunk_start + chunk_size
mask = (offs < chunk_end) & (offs < group_size)
```
