# 002 VAE Conv2D Residual Block

## Description

Complete fused residual block combining two sequential
`Conv3x3 -> GroupNorm -> SiLU` operations with residual addition.

This is the fundamental building block of Sana's VAE encoder/decoder:

`input -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> add(input)`

The block processes feature maps with the following fixed model dimensions:

- `channels = 256`
- `num_groups = 32`
- `kernel_size = 3`

## Inputs

| Name | Shape | Dtype |
| --- | --- | --- |
| `x` | `[batch_size, channels, height, width]` | `float32` |
| `conv1_weight` | `[channels, channels, kernel_size, kernel_size]` | `float32` |
| `norm1_weight` | `[channels]` | `float32` |
| `norm1_bias` | `[channels]` | `float32` |
| `conv2_weight` | `[channels, channels, kernel_size, kernel_size]` | `float32` |
| `norm2_weight` | `[channels]` | `float32` |
| `norm2_bias` | `[channels]` | `float32` |
| `eps` | `scalar` | `float32` |

## Outputs

| Name | Shape | Dtype |
| --- | --- | --- |
| `output` | `[batch_size, channels, height, width]` | `float32` |

## Benchmark Workloads

The benchmark suite covers the following workload shapes. Only the workload
definitions are listed here; latency, baseline, speedup, and SOL score values
are intentionally omitted.

| Workload |
| --- |
| `batch_size=16, height=64, width=64` |
| `batch_size=1, height=128, width=128` |
| `batch_size=1, height=131, width=131` |
| `batch_size=32, height=128, width=128` |
| `batch_size=2, height=128, width=128` |
| `batch_size=4, height=128, width=128` |
| `batch_size=2, height=256, width=256` |
| `batch_size=4, height=64, width=64` |
| `batch_size=64, height=64, width=64` |
| `batch_size=2, height=64, width=64` |
| `batch_size=1, height=1024, width=1024` |
| `batch_size=1, height=293, width=293` |
| `batch_size=4, height=256, width=256` |
| `batch_size=32, height=64, width=64` |
| `batch_size=8, height=64, width=64` |
| `batch_size=1, height=768, width=768` |
| `batch_size=4, height=128, width=96` |
| `batch_size=4, height=96, width=128` |
| `batch_size=8, height=64, width=128` |
| `batch_size=2, height=256, width=192` |

## Repository Layout

- `00_torch_ref/submission.py`: PyTorch reference implementation
- `01_triton/submission.py`: Triton implementation
- `main.py`: local correctness test driver
