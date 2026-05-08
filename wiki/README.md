---
title: "Knowledge Base — Open LLM Architectures"
tags: [readme]
updated: 2026-05-02
---

# Open LLM Architectures Knowledge Base

A research wiki tracking the lineage of modern open-weight large language models, from LLaMA (2023) onward.

## Layout

- `sources/` — one summary per ingested paper / article / talk. Each has frontmatter pointing back to the raw file in `../raw/`.
- `concepts/` — canonical articles for ideas referenced across multiple sources (architectures, training tricks, scaling laws, eval benchmarks).
- `people/` — researchers and labs worth tracking.
- `threads/` — synthesized deep-dives that cut across multiple sources.
- `assets/` — images embedded in wiki articles.

## Where to start

- [index](index.md) — the full index of sources, concepts, threads, people
- [sources/llama1-2023](sources/llama1-2023.md) — start here for the foundational open-weight architecture

## Conventions

- Internal navigation uses **standard relative markdown links** (e.g. `[concepts/rope](concepts/rope.md)`). These render correctly in both Obsidian and GitHub.
- Sources are slugged `name-year` (e.g. `llama1-2023`, `deepseek-v3-2024`).
- All claims are faithful to source material; no fabricated numbers or citations.
- Math is rendered with LaTeX *and* explained in words.
