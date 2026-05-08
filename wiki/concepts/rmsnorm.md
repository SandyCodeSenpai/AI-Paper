---
title: "RMSNorm"
tags: [transformer, normalization]
updated: 2026-05-02
aliases: [Root Mean Square Layer Normalization, RMS Norm]
---

# RMSNorm

## What it is

A simplified replacement for LayerNorm. Where LayerNorm subtracts the mean and divides by the standard deviation, RMSNorm just divides by the root-mean-square of the input — no mean-centering, no bias term.

```
LayerNorm(x) = γ · (x − μ) / σ + β
RMSNorm(x)  = γ · x / RMS(x)        where RMS(x) = sqrt(mean(x²) + ε)
```

Introduced in Zhang & Sennrich 2019.

## Why it matters

- **Cheaper.** ~7–64% faster than LayerNorm in their measurements; one fewer reduction.
- **Equivalent or better quality** at scale.
- **Re-centering invariance was never doing much** in modern Transformers, only the rescaling was load-bearing.

## Where it's used

- [[sources/llama1-2023]] introduced it as part of the modern open-weight stack
- Adopted by essentially all Llama-family models, Mistral, Mixtral, DeepSeek, Qwen, Gemma, Phi
- Often paired with [[concepts/pre-normalization]]

## See also

- [[concepts/pre-normalization]] — *where* the norm is applied (input vs. output of sublayer)
- LayerNorm (Ba et al. 2016) — what RMSNorm replaces
