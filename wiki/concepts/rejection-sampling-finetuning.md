---
title: "Rejection Sampling Fine-Tuning"
tags: [alignment, post-training, rlhf]
updated: 2026-05-07
aliases: [Rejection Sampling, RSFT, Best-of-N fine-tuning]
---

# Rejection Sampling Fine-Tuning

## What it is

A simpler alternative to [PPO](ppo.md) for RL fine-tuning. Procedure:

1. For each prompt in your training set, **sample K candidate generations** from the current policy (different temperatures, different seeds).
2. Score each candidate with a [reward model](reward-modeling.md).
3. **Keep only the top-scoring candidate** per prompt.
4. **Fine-tune the model on those top samples** with a standard SFT objective.

That's it. No policy gradient, no KL penalty, no value head. The exploration happens *outside* the gradient update — entirely via sampling.

## Why it works

- High-temperature sampling lets the model find responses better than its own greedy output.
- The reward model picks the best of K — implicitly enforcing the preference.
- Fine-tuning on those top samples shifts the policy toward higher-reward regions without the instability of policy gradients.

The delta between `max-of-K reward` and `median-of-K reward` ([sources/llama2-2023](../sources/llama2-2023.md) Figure 7) is the available improvement headroom — and it grows with K.

## Used in [sources/llama2-2023](../sources/llama2-2023.md)

- **RLHF V1–V4** trained entirely with rejection sampling — no PPO until V5.
- They sample many candidates from the largest model (70B), select with reward models, then fine-tune *all* sizes on those samples — so smaller models get **distilled** from the largest model's RM-curated outputs.
- Optimal sampling temperature shifts after each RL iteration: T=1 for SFT, T=1.2–1.3 for V5 RLHF model.

## Variants

- **Best-of-N at inference** is the same idea but applied at decode time, not as a fine-tuning signal. Useful for evaluation but doesn't compound.
- **Iterated Best-of-N** — one round of RSFT, then sample again from the new policy and repeat. Llama 2 V1 → V2 → V3 → V4 is a sequence of these.
- **Statistical rejection sampling** in Anthropic's older work — the same flavor, just framed as importance-sampling.

## Tradeoffs vs PPO

| | RSFT | PPO |
|---|---|---|
| Compute | Generation-heavy, training is plain SFT | Generation + value head + KL penalty |
| Memory | 1 model copy at training time | 4 model copies (policy, ref, RM, value) |
| Stability | Very stable; SFT-like | Sensitive to KL coef, clip, LR |
| Sample efficiency | Lower (most samples discarded) | Higher (every sample contributes) |
| Best for | Most of training, especially early | Fine-grained final polishing |

Llama 2's mixed approach — RSFT for V1–V4, RSFT+PPO for V5 — exploits both regimes.

## Where it's used

- [sources/llama2-2023](../sources/llama2-2023.md) — extensively documented, V1–V4
- Anthropic Claude post-training (described in Bai 2022)
- Many open chat-tune efforts that want RLHF benefits without PPO complexity
- DeepSeek-V3 and Kimi K2 use RSFT-style stages in their pipelines

## See also

- [concepts/rlhf](rlhf.md) · [concepts/ppo](ppo.md) · [concepts/reward-modeling](reward-modeling.md)
- DPO — RM-free alternative; collapses the RM and the policy update into one loss
