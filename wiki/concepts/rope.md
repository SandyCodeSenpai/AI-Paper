---
title: "Rotary Positional Embeddings (RoPE)"
tags: [transformer, positional-encoding, attention]
updated: 2026-05-02
aliases: [Rotary Position Embedding, RoPE]
---

# Rotary Positional Embeddings (RoPE)

## What it is

A way to encode token position into a Transformer by **rotating** query and key vectors at every attention layer, rather than adding a positional vector to the input embeddings. Introduced in Su et al. 2021 ("RoFormer"), now used in essentially every modern open LLM.

## Intuition

Pair up the Q and K vector dimensions into 2D blocks. For position `m`, rotate each 2D block by an angle `m · θ_i`, where `θ_i` differs per dimension pair (geometric series of frequencies, similar to sinusoidal embeddings).

When you compute the attention dot product `q_m · k_n`, the rotation cancels in a way that **the result depends only on the relative offset `m − n`**, not the absolute positions. So attention is naturally relative-position-aware without bolting on a separate relative bias.

## Why it matters

- **Relative-position semantics for free.** No need for explicit relative-position biases (T5-style) or learned embeddings.
- **Better long-context generalization.** Models extrapolate to longer contexts more gracefully than learned absolute positions, especially with positional interpolation / NTK-aware tweaks.
- **Cheap.** Just a per-layer pre-multiply of Q and K.
- **Compatible with KV cache.** Rotation is a function of position alone, so cached K vectors stay valid.

## Where it's used

- [sources/llama1-2023](../sources/llama1-2023.md) — first major open LLM to standardize RoPE
- All Llama family (Llama 2, 3, 4)
- Mistral, Mixtral, DeepSeek-V2/V3, Qwen, Gemma, Phi, OLMo, Kimi K2 — essentially all post-2023 open LLMs

## See also

- [concepts/pre-normalization](pre-normalization.md) · [concepts/rmsnorm](rmsnorm.md) · [concepts/swiglu](swiglu.md) — the rest of the modern open-LLM stack
- Sinusoidal positional encodings (Vaswani 2017) — the predecessor RoPE replaces
