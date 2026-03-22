# 030 Attention Output Projection with Residual

## Description

Fused attention output projection (o_proj) with residual addition. After computing attention outputs, this operation performs the final linear projection and adds the residual connection. The projection is a matmul (hidden_size x hidden_size) fused with elementwise residual add to eliminate intermediate memory traffic.

## Inputs

| Name | Shape | Dtype |
| --- | --- | --- |
| `attn_output` | `[batch_size, seq_len, hidden_size]` | `bfloat16` |
| `residual` | `[batch_size, seq_len, hidden_size]` | `bfloat16` |
| `o_proj_weight` | `[hidden_size, hidden_size]` | `bfloat16` |

## Outputs

| Name | Shape | Dtype |
| --- | --- | --- |
| `output` | `[batch_size, seq_len, hidden_size]` | `bfloat16` |

## Benchmark Workloads

The benchmark suite covers the following workload shapes. Only the workload
definitions are listed here; latency, baseline, speedup, and SOL score values
are intentionally omitted.

| Workload |
| --- |
| `batch_size=16, seq_len=512` |
| `batch_size=4, seq_len=128` |
| `batch_size=8, seq_len=1024` |
| `batch_size=1, seq_len=1571` |
| `batch_size=4, seq_len=1024` |
| `batch_size=2, seq_len=2053` |
| `batch_size=8, seq_len=997` |
| `batch_size=16, seq_len=256` |
| `batch_size=64, seq_len=128` |
| `batch_size=32, seq_len=256` |
| `batch_size=8, seq_len=512` |
| `batch_size=1, seq_len=1024` |
| `batch_size=16, seq_len=128` |
| `batch_size=2, seq_len=293` |
| `batch_size=1, seq_len=2048` |
| `batch_size=1, seq_len=256` |
