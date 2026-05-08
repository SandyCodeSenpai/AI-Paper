---
title: "Pre-normalization"
tags: [transformer, normalization, training-stability]
updated: 2026-05-02
aliases: [Pre-LN, Pre-norm]
---

# Pre-normalization (Pre-LN)

## What it is

A choice of *where* to apply layer normalization inside a Transformer block. Two options:

**Post-LN (original Vaswani 2017):**
```
x' = LayerNorm(x + Sublayer(x))
```

**Pre-LN (modern):**
```
x' = x + Sublayer(LayerNorm(x))
```

Pre-LN moves the norm to the **input** of each sublayer, leaving the residual stream unnormalized.

## Why it matters

- **Training stability at depth and scale.** Post-LN models suffer gradient pathologies as depth grows; learning-rate warmup and careful initialization are required to train them. Pre-LN is dramatically more forgiving.
- **No warmup needed** in many cases (or much shorter schedules).
- The residual stream stays "clean" — gradients flow through the residual path without being squashed by a norm at each step.

The tradeoff is a small quality cost vs. a well-tuned post-LN, but the training stability is so much better that nobody does post-LN at modern scale.

## Where it's used

- Originally popularized by GPT-2 / GPT-3
- [[sources/llama1-2023]] adopted it as part of the standard recipe (paired with [[concepts/rmsnorm]])
- Universal in modern open LLMs

## See also

- [[concepts/rmsnorm]] — *what* normalization is applied
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) — the analysis paper
