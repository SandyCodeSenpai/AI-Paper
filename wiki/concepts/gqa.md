---
title: "Grouped-Query Attention (GQA)"
tags: [transformer, attention, kv-cache, inference]
updated: 2026-05-07
aliases: [GQA, Grouped Query Attention]
---

# Grouped-Query Attention (GQA)

## What it is

A multi-head attention variant where **K and V projections are shared across groups of query heads**, instead of one-per-head (MHA) or one-shared (MQA).

```
MHA  : n_heads Q projections,  n_heads   K/V projections        (most flexible, most memory)
MQA  : n_heads Q projections,  1         K/V projection         (cheapest, often regresses)
GQA  : n_heads Q projections,  n_kv_heads K/V projections       (compromise)
       where n_kv_heads divides n_heads, group_size = n_heads / n_kv_heads
```

Introduced in Ainslie et al. 2023; adopted at scale by [[sources/llama2-2023]] for the 34B and 70B models.

## Why it matters

The **KV cache** stored at inference is `n_layers × n_kv_heads × seq_len × head_dim` per request. For long context and large batch, this dominates GPU memory.

A 70B model in fp16 with full MHA, KV cache for one request at 4K tokens: ~10 GB. With GQA at 8 KV heads (group size 8): ~1.25 GB — an 8× saving. The savings compound at long context and large batch.

Quality cost is small. From the Llama 2 ablation (Table 18, 30B / 150B tokens, MMLU 5-shot): MHA 28.0, **GQA 26.9**, MQA 14.5. GQA matches MHA on most tasks and dramatically beats MQA on knowledge-heavy benchmarks.

## Compensating for parameter loss

Shrinking n_kv_heads removes parameters from the attention block. To keep total parameter count comparable, the FFN dimension is bumped up — Llama 2 uses ~1.3× FFN expansion when going to GQA.

## Where it's used

- [[sources/llama2-2023]] — first major open release with GQA at scale (34B, 70B with 8 KV heads)
- All Llama 3 / Llama 4 models
- Mistral 7B, Mixtral, DeepSeek-V2/V3, Qwen 2/2.5/3, Gemma 2
- Default setting in the modern open-LLM stack

## See also

- [[concepts/rope]] · [[concepts/swiglu]] · [[concepts/rmsnorm]] — the rest of the LLaMA-recipe stack
- MQA (Shazeer 2019) — the predecessor GQA improves on
- FlashAttention / PagedAttention — orthogonal optimizations that further reduce KV-cache pain
