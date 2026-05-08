---
title: "Reward Modeling"
tags: [alignment, post-training, rlhf]
updated: 2026-05-07
aliases: [Reward Model, RM]
---

# Reward Modeling

## What it is

A reward model (RM) takes a (prompt, response) pair and outputs a scalar score that approximates **how well humans would rate the response**. It's the proxy that lets RL fine-tuning optimize for human preference without humans in the inner loop.

Architecturally: take a chat-finetuned LM checkpoint, **replace the next-token-prediction head with a scalar regression head** on the final hidden state of the response.

## Training: pairwise ranking loss

Given preference triples (prompt x, chosen response y_c, rejected y_r):

$$L = -\log \sigma(r_\theta(x, y_c) - r_\theta(x, y_r))$$

Equivalent to the Bradley-Terry / logistic-pairwise model. Llama 2 adds a margin term scaled by preference strength:

$$L = -\log \sigma(r_\theta(x, y_c) - r_\theta(x, y_r) - m(r))$$

where m(r) is larger for "significantly better" annotations, smaller for "negligibly better." Improves accuracy on clearly-separable pairs without hurting accuracy on close calls.

## Key design decisions ([[sources/llama2-2023]])

- **Init from chat checkpoint, not base.** The RM "knows what the model knows," preventing reward hallucinations where the RM rewards confidently-wrong responses because it can't tell they're wrong.
- **Two RMs for helpfulness vs safety.** They conflict, and a single RM that's good at both is hard to train. Combined piecewise at PPO time (safety overrides if `IS_SAFETY(prompt)` or `R_safety < threshold`).
- **Mix open-source preference data with internal data.** Open data prevents overfitting and reward hacking; internal data targets the actual model distribution.
- **Iterative collection.** Each new RLHF version's outputs become the source of preference data for the next RM update — keeps RM in distribution.
- **Separate held-out test sets** ("Meta Helpful," "Meta Safety") to track RM quality independent of the RL policy.

## Failure modes

- **Reward hacking** — policy finds outputs that score high on RM but humans hate. Symptoms: verbose, hedged, repetitive, or stylistically off.
- **Distribution shift** — RM trained on outputs from model V_n is asked to score outputs from V_{n+5}. Accuracy degrades. Mitigated by iterative collection.
- **Reward overfitting** — RM hits ~95% accuracy on training pairs but doesn't generalize. Mitigated by mixing in open-source preference data.
- **Helpfulness/safety tradeoff** — single RM over-refuses to be safe, or over-helps to be helpful. Mitigated by two-RM split.

## Reward model results from Llama 2 (Table 7)

Helpfulness RM 63.2 avg, Safety RM 64.3 avg — both beating GPT-4-as-judge (63.0). Larger RMs and more data continue to improve accuracy without saturation (Figure 6).

## Where it's used

- [[sources/llama2-2023]] — detailed two-RM design and ranking-loss-with-margin
- All [[concepts/rlhf|RLHF]]-trained models implicitly
- DPO / direct-preference-optimization variants *don't* train an explicit RM — they derive policy updates directly from preference data

## See also

- [[concepts/rlhf]] · [[concepts/ppo]] · [[concepts/rejection-sampling-finetuning]]
- DPO (Rafailov 2023) — the RM-free alternative
