---
title: "SwiGLU activation"
tags: [transformer, activation, ffn]
updated: 2026-05-02
aliases: [SwiGLU, Swish-Gated Linear Unit]
---

# SwiGLU

## What it is

A *gated* activation function used in the feedforward layer of a Transformer, introduced in Shazeer 2020 ("GLU Variants Improve Transformer"). Replaces the standard `ReLU(xW)` with:

```
SwiGLU(x) = Swish(xW) ⊙ (xV)
```

where:
- `Swish(z) = z · sigmoid(z)` (a smooth, non-monotonic activation)
- `⊙` is elementwise multiplication
- `W` and `V` are two separate learned projections

So the FFN now has **two** input projection matrices (W and V) instead of one. The "gate" branch (xV) modulates the activation branch (Swish(xW)) elementwise.

## Parameter-count adjustment

Because SwiGLU uses two input matrices, a naive (4d → 4d) FFN doubles the params. To keep param count comparable to a vanilla FFN, LLaMA and successors use **hidden dim ≈ (2/3) · 4d** instead of `4d`.

## Why it matters

- Empirically stronger than ReLU / GeLU in scaling-law experiments
- Smoothness of Swish + multiplicative gating gives the FFN more expressive capacity per param
- Now standard in essentially all modern open LLMs

## Where it's used

- [[sources/llama1-2023]] adopted it from PaLM
- All Llama family, Mistral, Mixtral, DeepSeek, Qwen, Gemma, Phi — universal in open LLMs

## See also

- [[concepts/rmsnorm]] · [[concepts/rope]] · [[concepts/pre-normalization]] — the rest of the LLaMA recipe
- ReLU, GeLU, GLU — the alternatives SwiGLU outperforms
