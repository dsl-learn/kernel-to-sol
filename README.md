# SOL-ExecBench Solutions

This repository collects my solutions and writeups for the
[NVIDIA SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench) benchmark.

## Goals

- Build a structured set of SOL-ExecBench solutions.
- Provide reproducible implementations with clear code comments.
- Document transferable GPU kernel optimization patterns.

## Writeup Structure

Each problem writeup will typically include:

- Problem understanding and constraints
- Baseline implementation
- Optimized versions (e.g., memory access, parallel strategy, fusion)
- Performance comparison and key takeaways

## Status

This repository is a work in progress and will be updated continuously.

## Problems

- [001_attn_bwd](./001_attn_bwd): Backward pass for attention softmax, dropout, and value matmul.
- [002_vae_conv2d](./002_vae_conv2d): Fused VAE residual block with Conv3x3, GroupNorm, SiLU, and residual addition.

## Reference

- Benchmark: <https://research.nvidia.com/benchmarks/sol-execbench>
