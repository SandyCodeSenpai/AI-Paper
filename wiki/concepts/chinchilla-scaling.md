---
title: "Chinchilla scaling laws"
tags: [scaling-laws, pretraining, compute]
updated: 2026-05-02
aliases: [Chinchilla, compute-optimal scaling]
---

# Chinchilla Scaling Laws

## What it is

Hoffmann et al. 2022 ("Training Compute-Optimal Large Language Models"), from DeepMind. Found that for a fixed *training* compute budget, the optimal allocation between model size N and tokens D is **roughly equal scaling**: doubling compute should roughly double both N and D. Earlier work (Kaplan 2020) had recommended scaling N much faster than D.

The empirical rule of thumb that came out of this: **~20 tokens per parameter** for compute-optimal training.

## Why it matters

- Kaplan-era models like GPT-3 (175B params, ~300B tokens) were *under-trained* per Chinchilla. Chinchilla itself was 70B trained on 1.4T tokens — same compute as Gopher (280B / 300B tokens) but a far better model.
- Reset the field's intuition that "bigger is better" — for a given compute budget, smaller-and-longer often wins.

## How LLaMA pushed past it

[sources/llama1-2023](../sources/llama1-2023.md) argued that Chinchilla optimizes the wrong objective for *production*: training compute is one-shot, but **inference** compute is paid forever per query. So they trained models *past* Chinchilla-optimal — LLaMA-7B on 1T tokens (≈140 tokens/param), well beyond the 20-tokens-per-param rule, because the resulting model is much cheaper to serve.

This reframing — inference-optimal vs. compute-optimal — became standard practice. Modern small-but-strong models (Mistral 7B, Phi-3, Gemma 2) lean even harder into it.

## See also

- [sources/llama1-2023](../sources/llama1-2023.md) — first paper to push past Chinchilla-optimal in service of openness + inference cost
- Kaplan et al. 2020 — the earlier scaling law Chinchilla refined
- Phi-3 — extends the data-quality dimension of the same argument
