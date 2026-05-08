---
title: "Reinforcement Learning from Human Feedback (RLHF)"
tags: [alignment, post-training, rl]
updated: 2026-05-07
aliases: [RLHF]
---

# Reinforcement Learning from Human Feedback (RLHF)

## What it is

A multi-stage pipeline for aligning a language model with human preferences:

```
1. SFT      : fine-tune base model on curated prompt-response pairs
2. Preference data: collect (prompt, chosen, rejected) triples from human annotators
3. Reward model: train a scalar regression head to predict the chosen-vs-rejected preference
4. RL fine-tuning: optimize the policy (the LM) to maximize the reward model's score,
                   penalized by KL divergence to the SFT model
```

Originated in InstructGPT (Ouyang 2022). [sources/llama2-2023](../sources/llama2-2023.md) gave the first detailed open-paper account of running it end-to-end at scale.

## Components

- **[SFT](sft.md)** — gives the model a starting policy that already produces dialogue-shaped responses.
- **Preference data** — humans pick "A vs B" between two model outputs. Often with a graded scale (significantly / better / slightly / negligibly better).
- **[Reward Model](reward-modeling.md)** — a regression head fine-tuned from a chat checkpoint, predicting which response humans preferred. Usually trained with a binary ranking loss.
- **RL algorithm** — typically [PPO](ppo.md) (Schulman 2017) or [Rejection Sampling fine-tuning](rejection-sampling-finetuning.md) (Bai 2022). DPO (Rafailov 2023) bypasses the explicit RM and policy-gradient step.

## Key design decisions

- **Two reward models vs one.** Llama 2 uses separate Helpfulness and Safety RMs because the two objectives conflict. Combined piecewise at PPO time.
- **KL penalty (β) tuning.** Too low: reward hacking; too high: model can't move from SFT. Llama 2 uses β=0.01 for 7B/13B, β=0.005 for 34B/70B.
- **Iterative collection.** Each new RLHF version becomes the source of preference data for the next round — keeps RM in distribution.
- **Initialization of RM.** From a chat checkpoint, not from base — RM "knows what the model knows," prevents reward hallucinations.

## Variants and successors

- **Constitutional AI / RLAIF** (Anthropic, Bai 2022) — replaces human preference labels with model-generated preferences against a written constitution.
- **DPO (Direct Preference Optimization)** (Rafailov 2023) — derives a closed-form policy update from preference data, skipping the RM entirely. Simpler, often comparable.
- **GRPO** (DeepSeek-Math 2024; later DeepSeek-R1) — group-relative advantage estimation, more stable than PPO for reasoning tasks.

## Where it's used

- [sources/llama2-2023](../sources/llama2-2023.md) — the canonical open description
- Implicitly: ChatGPT, Claude, Gemini, all major proprietary chat models
- Open: Zephyr (DPO), Tülu, Qwen-Chat, DeepSeek-V2-Chat, Llama 3-Instruct, etc.

## See also

- [concepts/sft](sft.md) · [concepts/reward-modeling](reward-modeling.md) · [concepts/ppo](ppo.md) · [concepts/rejection-sampling-finetuning](rejection-sampling-finetuning.md)
- Constitutional AI · DPO · GRPO (alternative alignment recipes)
