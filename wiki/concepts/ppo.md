---
title: "Proximal Policy Optimization (PPO)"
tags: [rl, alignment, optimization]
updated: 2026-05-07
aliases: [PPO]
---

# Proximal Policy Optimization (PPO)

## What it is

A policy-gradient RL algorithm (Schulman et al. 2017) that optimizes a stochastic policy with a **clipped surrogate objective**, preventing destructively large updates while still being sample-efficient. The default RL algorithm for [[concepts/rlhf|RLHF]] in language models.

## The clipped objective

For each (state, action) under the old policy π_old, define the importance ratio:

$$\rho_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{old}(a_t | s_t)}$$

PPO maximizes:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[\min(\rho_t \hat{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$

where $\hat{A}_t$ is an advantage estimate. The clip (`ε ≈ 0.2`) prevents updates from moving the policy too far from the old one in a single step — that's the "proximal" part.

## In RLHF for LLMs

State = prompt + tokens generated so far. Action = next token. The advantage is computed per token using the reward model's score on the full generation:

$$R(g | p) = \tilde{R}_c(g | p) - \beta \, D_{KL}(\pi_\theta(g|p) \,\|\, \pi_0(g|p))$$

The **KL penalty** to the SFT model (π_0) is the second guardrail — alongside the PPO clip — preventing reward hacking by keeping the policy close to its starting point.

[[sources/llama2-2023]] PPO settings:
- KL coefficient β = 0.01 (7B/13B), 0.005 (34B/70B)
- PPO clip ε = 0.2
- Mini-batch 64, batch 512, 1 grad step per mini-batch
- Constant LR 1e-6, AdamW (β₁=0.9, β₂=0.95)
- 200–400 PPO iterations per RLHF version

## Cost

PPO requires **four model copies** in memory: policy, reference (frozen SFT), reward model, value head. For a 70B model this is brutal. Llama 2 mitigates with FSDP sharding, with one wrinkle: weights are *consolidated* to each node before generation (full FSDP sharding makes generation ~20× slower).

## Alternatives

- **[[concepts/rejection-sampling-finetuning|Rejection Sampling fine-tuning]]** (cheaper, no policy gradient — Llama 2 used it for V1–V4)
- **DPO** (Rafailov 2023) — closed-form policy update from preferences, no separate RM, no PPO loop
- **GRPO** (DeepSeek-Math / [[sources/deepseek-r1-2025]]) — group-relative advantage; more stable than PPO for long reasoning chains
- **REINFORCE-style baselines** — simpler, occasionally competitive on language tasks

## Where it's used

- [[sources/llama2-2023]] — V5 only (after rejection-sampling fine-tuning V1–V4)
- InstructGPT — original RLHF + PPO recipe
- Many open chat models pre-DPO

## See also

- [[concepts/rlhf]] · [[concepts/reward-modeling]] · [[concepts/rejection-sampling-finetuning]]
- TRPO (Schulman 2015) — PPO's predecessor with explicit trust region
- GRPO — DeepSeek's PPO replacement for reasoning RL
