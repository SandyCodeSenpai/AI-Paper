---
title: "Ghost Attention (GAtt)"
tags: [post-training, multi-turn, attention]
updated: 2026-05-07
aliases: [GAtt, Ghost Attention]
---

# Ghost Attention (GAtt)

## What it is

A training-data trick (no architectural change) introduced in [[sources/llama2-2023]] §3.3 to fix multi-turn instruction forgetting.

**Problem:** In a dialogue with a system instruction like "always answer with emojis," the chat model obeys at turn 1 but forgets by turn 3 — the instruction is far away in the context, and the model's attention drifts to the immediate user message.

**Fix:** During fine-tuning on multi-turn dialogues, **synthetically prepend the system instruction to *every* user turn**. At training time, the instruction is everywhere — every turn provides supervision for keeping it salient. At inference time, you only put the instruction once at the actual system message; the model has learned to keep attending to it.

## The training-data construction

Given multi-turn dialogue `[u₁, a₁, u₂, a₂, …, uₙ, aₙ]` and instruction `inst`:

1. Replace each user turn `uᵢ` with `concat(inst, uᵢ)` to form augmented data.
2. Use the *latest* RLHF model to sample assistant responses that comply with `inst` throughout (Rejection Sampling style).
3. **Zero out the loss on all assistant tokens except the last turn `aₙ`** — otherwise the model overfits to seeing `inst` repeated, which is a synthetic-only artifact.

The instruction list at training time is sampled from synthetic constraints: hobbies ("you enjoy tennis"), language ("speak in French"), public figure ("act as Napoleon"), etc.

## Why it works

By design, the model can only get the loss right on `aₙ` if attention to the system instruction stays alive throughout the dialogue. Figure 10 in the paper shows attention activations to the system message stay high across all dialogue turns *after* GAtt training, where the baseline model's attention to it decays sharply.

## Caveats

- Requires the model already follows single-turn system instructions (so GAtt is applied **after** RLHF V3 in Llama 2).
- The synthetic prepended instructions are an artifact of training data; if not handled (loss masking on intermediates), the model can latch onto the prepending pattern itself.
- Currently a "vanilla" implementation per the paper — the authors note further iteration could improve it (e.g., teaching the model to *change* the system message mid-conversation).

## Where it's used

- [[sources/llama2-2023]] — the introducing paper
- Subsequent open-chat training pipelines have largely adopted it or close variants

## See also

- [[concepts/sft]] · [[concepts/rejection-sampling-finetuning]] — the surrounding pipeline
- Context Distillation (Askell 2021) — another "preprompt as training data" technique
