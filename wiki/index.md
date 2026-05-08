---
title: "Wiki Index"
tags: [index]
updated: 2026-05-07
---

# Wiki Index

Top-level map of the knowledge base. Auto-maintained — when new sources or concepts are added, this file is updated.

## Sources (papers, articles, talks)

### Open LLM architectures
- [sources/llama1-2023](sources/llama1-2023.md) — LLaMA: Open and Efficient Foundation Language Models (Meta, Feb 2023)
- [sources/llama2-2023](sources/llama2-2023.md) — Llama 2: Open Foundation and Fine-Tuned Chat Models (Meta, Jul 2023)

### Pending ingest
- raw/papers/mistral7b-2023.pdf
- raw/papers/mixtral-2024.pdf
- raw/papers/olmo-2024.pdf
- raw/papers/phi3-2024.pdf
- raw/papers/deepseek-v2-2024.pdf
- raw/papers/llama3-2024.pdf
- raw/papers/gemma2-2024.pdf
- raw/papers/qwen2.5-2024.pdf
- raw/papers/deepseek-v3-2024.pdf
- raw/papers/deepseek-r1-2025.pdf
- raw/papers/qwen3-2025.pdf
- raw/papers/kimi-k2-2025.pdf
- *(missing PDFs: DBRX, Snowflake Arctic, Llama 4 — non-arXiv, awaiting decision)*

## Concepts

### Architecture (decoder-only Transformer recipe)
- [concepts/pre-normalization](concepts/pre-normalization.md) — where to apply LayerNorm inside a block
- [concepts/rmsnorm](concepts/rmsnorm.md) — simplified LayerNorm without mean-centering
- [concepts/swiglu](concepts/swiglu.md) — gated activation in the FFN
- [concepts/rope](concepts/rope.md) — rotary positional embeddings
- [concepts/gqa](concepts/gqa.md) — grouped-query attention; KV-cache savings without MQA's quality cost

### Pretraining
- [concepts/chinchilla-scaling](concepts/chinchilla-scaling.md) — compute-optimal data/model allocation, and why LLaMA pushed past it

### Post-training / alignment
- [concepts/sft](concepts/sft.md) — supervised fine-tuning; "quality is all you need"
- [concepts/rlhf](concepts/rlhf.md) — RLHF pipeline overview
- [concepts/reward-modeling](concepts/reward-modeling.md) — RM training, two-RM design, ranking-with-margin loss
- [concepts/ppo](concepts/ppo.md) — Proximal Policy Optimization in the LLM RLHF context
- [concepts/rejection-sampling-finetuning](concepts/rejection-sampling-finetuning.md) — best-of-N as a training signal
- [concepts/ghost-attention](concepts/ghost-attention.md) — multi-turn instruction-following data trick

## Threads
*(none yet — synthesized cross-source deep-dives go here)*

## People
*(none yet — researcher / lab pages go here)*

## Outputs

### Notebooks
- `outputs/llama1-2023-notebook.ipynb` — RMSNorm/SwiGLU/RoPE in PyTorch + Llama 1 results visualizations
- `outputs/llama2-2023-notebook.ipynb` — GQA implementation + KV-cache analysis + RLHF visualizations
