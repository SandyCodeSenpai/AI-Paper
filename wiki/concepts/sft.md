---
title: "Supervised Fine-Tuning (SFT)"
tags: [post-training, instruction-tuning]
updated: 2026-05-07
aliases: [Supervised Fine-Tuning, SFT, Instruction Tuning]
---

# Supervised Fine-Tuning (SFT)

## What it is

Fine-tune a pretrained base LM on a curated set of (prompt, response) pairs using the standard autoregressive objective, with **loss masked on the prompt** so the model only updates on the response tokens.

Result: a model that follows instructions in dialogue format. SFT is the bridge between a raw base model (good at next-token prediction) and an aligned chat model.

## Standard recipe

- Format: `<prompt><sep><response>` with a separator token.
- Loss: cross-entropy on response tokens only (zero out user/prompt tokens in the loss).
- LR: ~1–2× lower than pretraining peak LR (e.g., 2e-5 for Llama 2-Chat).
- Batch: small, 64-256 examples.
- Epochs: 2–3 typically; risk of overfitting beyond that.

## "Quality is all you need"

The most influential finding from [[sources/llama2-2023]] §3.1 (echoing Zhou et al. 2023, "LIMA"): **the size of the SFT dataset matters less than its quality**. Llama 2 abandoned millions of third-party instruction examples and ended up with **27,540 vendor-written prompt+response pairs**, where the same person wrote both halves. Result: better SFT model than from millions of crowdsourced ones.

This finding has become received wisdom — Phi, Zephyr, Tülu, and many others follow the "small high-quality SFT set" pattern.

## Why mask the prompt

If the loss includes the prompt, the model spends gradient on learning to *generate* user prompts as well as responses. That's not what we want — we want the model to be good at responding given a prompt.

In multi-turn dialogues, mask all turns except the assistant turn(s) you're training on. [[concepts/ghost-attention|Ghost Attention]] takes this further: zero out loss on intermediate assistant turns to avoid learning the synthetic per-turn system prompt.

## What SFT alone gets you

A polite, helpful, dialogue-formatted model — but with sharp limitations:
- Still mimics annotator style; capped by the writing ability of your annotators.
- Doesn't learn from preferences (only from positive examples).
- Doesn't see what *bad* responses look like, so doesn't reliably refuse them.

[[concepts/rlhf|RLHF]] is what closes those gaps.

## Where it's used

- [[sources/llama2-2023]] — the canonical "quality > quantity" SFT recipe
- Every modern aligned LLM has an SFT stage as the first step before RLHF/DPO
- Phi-3, Qwen-Chat, Gemma-Instruct, DeepSeek-Chat — all start with SFT

## See also

- [[concepts/rlhf]] — what comes after SFT
- [[concepts/ghost-attention]] — multi-turn SFT trick
- LIMA (Zhou et al. 2023) — the canonical citation for "quality > quantity"
