# 001 Attention Backward

## Description

Backward pass for attention softmax, dropout, and value matmul.
It computes gradients through the following stages:

`transpose -> batched matmul -> dropout -> softmax -> GQA head expansion`

In this problem, the reference implementation uses the following fixed model
dimensions:

- `num_attention_heads = 80`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `attention_dropout = 0.1`

## Inputs

| Name | Shape | Dtype |
| --- | --- | --- |
| `grad_attn_output` | `[batch_size, seq_len_q, num_attention_heads, head_dim]` | `bfloat16` |
| `attn_weights` | `[batch_size, num_attention_heads, seq_len_q, seq_len_kv]` | `bfloat16` |
| `attn_weights_dropped` | `[batch_size, num_attention_heads, seq_len_q, seq_len_kv]` | `bfloat16` |
| `value_states` | `[batch_size, num_key_value_heads, seq_len_kv, head_dim]` | `bfloat16` |
| `dropout_mask` | `[batch_size, num_attention_heads, seq_len_q, seq_len_kv]` | `bool` |
| `attention_dropout` | `scalar` | `float32` |

## Outputs

| Name | Shape | Dtype |
| --- | --- | --- |
| `grad_attn_scores` | `[batch_size, num_attention_heads, seq_len_q, seq_len_kv]` | `bfloat16` |
| `grad_value_states` | `[batch_size, num_key_value_heads, seq_len_kv, head_dim]` | `bfloat16` |

## Benchmark Workloads

The benchmark suite covers the following workload shapes. Only the workload
definitions are listed here; latency, baseline, speedup, and SOL score values
are intentionally omitted.

| Workload |
| --- |
| `batch_size=4, seq_len_q=256, seq_len_kv=256` |
| `batch_size=8, seq_len_q=373, seq_len_kv=449` |
| `batch_size=4, seq_len_q=1024, seq_len_kv=2048` |
| `batch_size=64, seq_len_q=128, seq_len_kv=128` |
| `batch_size=2, seq_len_q=256, seq_len_kv=512` |
| `batch_size=32, seq_len_q=691, seq_len_kv=773` |
| `batch_size=8, seq_len_q=128, seq_len_kv=128` |
| `batch_size=32, seq_len_q=512, seq_len_kv=512` |
| `batch_size=4, seq_len_q=211, seq_len_kv=293` |
| `batch_size=8, seq_len_q=256, seq_len_kv=256` |
| `batch_size=16, seq_len_q=128, seq_len_kv=256` |
| `batch_size=1, seq_len_q=1024, seq_len_kv=1024` |
| `batch_size=16, seq_len_q=256, seq_len_kv=512` |
| `batch_size=32, seq_len_q=128, seq_len_kv=128` |
| `batch_size=1, seq_len_q=512, seq_len_kv=512` |
| `batch_size=1, seq_len_q=4096, seq_len_kv=4096` |

## Repository Layout

- `00_torch_ref/submission.py`: PyTorch reference implementation
- `01_triton/submission.py`: Triton implementation
- `main.py`: local correctness test driver
