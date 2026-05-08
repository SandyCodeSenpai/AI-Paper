"""Build outputs/llama2-2023-notebook.ipynb from scratch using stdlib json only.

Run: python3 tools/build_llama2_notebook.py

The notebook walks through the Llama 2 paper (Touvron et al., 2023):
  1. Implements GQA (Grouped-Query Attention) in PyTorch and compares
     KV cache memory cost against MHA and MQA
  2. Reproduces the parameter counts for the 4 sizes (7B/13B/34B/70B),
     including the GQA + FFN-bump trick that preserves param count
  3. Walks through the RLHF math (ranking loss with margin, PPO objective)
  4. Visualizes the headline results — base-model gains over LLaMA 1,
     win-rates vs ChatGPT, safety violations, RLHF V1->V5 progression

To run the notebook itself, install: torch, matplotlib, numpy, jupyter.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "llama2-2023-notebook.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _split(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(lines),
    }


def _split(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + ("\n" if i < len(parts) - 1 else "") for i, p in enumerate(parts)]


cells = []

cells.append(md(
    "# Llama 2 Paper Walkthrough — GQA, RLHF, Results",
    "",
    "Companion notebook to **Llama 2: Open Foundation and Fine-Tuned Chat Models** "
    "(Touvron et al., GenAI Meta, July 2023, [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)).",
    "",
    "Linked notes: `wiki/sources/llama2-2023.md` and concept articles "
    "`wiki/concepts/{gqa, rlhf, reward-modeling, ppo, rejection-sampling-finetuning, sft, ghost-attention}.md`.",
    "",
    "**What this notebook does:**",
    "1. Implements **Grouped-Query Attention (GQA)** in PyTorch and compares KV-cache memory against MHA and MQA",
    "2. Reproduces the parameter counts for the four Llama 2 sizes — including the **GQA + FFN-bump trick** that preserves total params",
    "3. Walks through the **RLHF math** — ranking loss with margin (reward model), PPO objective with KL penalty",
    "4. Visualizes the headline results — base-model gains over LLaMA 1, win-rates vs ChatGPT, safety violations, RLHF V1→V5 progression",
    "",
    "**To run:** `pip install torch matplotlib numpy jupyter`",
))

cells.append(md("## 0. Setup"))

cells.append(code(
    "import math",
    "import torch",
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import matplotlib.pyplot as plt",
    "import numpy as np",
    "",
    "torch.manual_seed(0)",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "print(f'PyTorch {torch.__version__} on {device}')",
))

# =========================================================================
# 1. GQA
# =========================================================================
cells.append(md(
    "## 1. Grouped-Query Attention (GQA)",
    "",
    "The **only architectural change** from LLaMA 1. Instead of one K and V projection per attention head (MHA), "
    "K and V are shared across **groups** of query heads.",
    "",
    "```",
    "MHA  : n_heads Q,  n_heads   K/V  (most flexible, most memory)",
    "MQA  : n_heads Q,  1         K/V  (cheapest, often regresses on quality)",
    "GQA  : n_heads Q,  n_kv_heads K/V (compromise, group_size = n_heads / n_kv_heads)",
    "```",
    "",
    "**Llama 2-70B uses 64 Q heads and 8 KV heads → group size 8.** This shrinks the KV cache by 8× without "
    "the quality regressions of MQA.",
))

cells.append(code(
    "class GQAttention(nn.Module):",
    "    def __init__(self, dim: int, n_heads: int, n_kv_heads: int):",
    "        super().__init__()",
    "        assert n_heads % n_kv_heads == 0, 'n_heads must be divisible by n_kv_heads'",
    "        self.n_heads = n_heads",
    "        self.n_kv_heads = n_kv_heads",
    "        self.head_dim = dim // n_heads",
    "        self.repeats = n_heads // n_kv_heads  # how many Q heads share each KV head",
    "",
    "        # Q stays full-width; K and V shrink to n_kv_heads",
    "        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)",
    "        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)",
    "        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)",
    "        self.wo = nn.Linear(dim, dim, bias=False)",
    "",
    "    def forward(self, x):",
    "        bsz, seq_len, _ = x.shape",
    "        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)",
    "        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)",
    "        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)",
    "",
    "        # Repeat K and V across each group of Q heads",
    "        # k: [bsz, seq, n_kv_heads, head_dim] -> [bsz, seq, n_heads, head_dim]",
    "        k = k.repeat_interleave(self.repeats, dim=2)",
    "        v = v.repeat_interleave(self.repeats, dim=2)",
    "",
    "        q = q.transpose(1, 2)  # [bsz, n_heads, seq, head_dim]",
    "        k = k.transpose(1, 2)",
    "        v = v.transpose(1, 2)",
    "",
    "        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)",
    "        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()",
    "        scores = scores.masked_fill(mask, float('-inf'))",
    "        attn = F.softmax(scores, dim=-1)",
    "        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, -1)",
    "        return self.wo(out)",
    "",
    "# Smoke test",
    "dim, n_heads, n_kv_heads, seq_len = 512, 8, 2, 16",
    "gqa = GQAttention(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads)",
    "x = torch.randn(2, seq_len, dim)",
    "y = gqa(x)",
    "print(f'GQA forward: in {tuple(x.shape)} -> out {tuple(y.shape)}')",
    "print(f'  n_heads = {n_heads}, n_kv_heads = {n_kv_heads}, group_size = {n_heads // n_kv_heads}')",
    "print(f'  KV projection params: {gqa.wk.weight.numel() + gqa.wv.weight.numel():,}')",
    "print(f'  MHA-equivalent KV params would be: {2 * dim * dim:,} (8x more for n_kv_heads=8)')",
))

# 1.2 KV cache analysis
cells.append(md(
    "### 1.1 KV cache memory comparison",
    "",
    "The reason GQA matters is **inference memory**. The KV cache stored per request is "
    "`n_layers × n_kv_heads × seq_len × head_dim × 2 (K and V) × bytes`. For a Llama 2-70B at 4K context, "
    "this is the difference between fitting 1 request and fitting 8 on the same hardware.",
))

cells.append(code(
    "def kv_cache_bytes(n_layers, n_kv_heads, seq_len, head_dim, dtype_bytes=2):",
    "    \"\"\"Bytes for one request's KV cache. dtype_bytes=2 for fp16.\"\"\"",
    "    return n_layers * n_kv_heads * seq_len * head_dim * 2 * dtype_bytes  # 2 for K and V",
    "",
    "# Llama 2-70B: 80 layers, 64 Q heads, head_dim 128",
    "scenarios = [",
    "    ('MHA  (64 KV heads)', 80, 64, 128),",
    "    ('GQA  (8 KV heads, Llama 2-70B)', 80, 8, 128),",
    "    ('MQA  (1 KV head)', 80, 1, 128),",
    "]",
    "",
    "print(f\"{'Variant':<35}{'KV @ 4K context':>20}{'KV @ 32K context':>20}\")",
    "print('-' * 75)",
    "for name, n_layers, n_kv, hd in scenarios:",
    "    kv_4k = kv_cache_bytes(n_layers, n_kv, 4096, hd) / 1e9",
    "    kv_32k = kv_cache_bytes(n_layers, n_kv, 32768, hd) / 1e9",
    "    print(f'{name:<35}{kv_4k:>17.2f} GB{kv_32k:>17.2f} GB')",
    "",
    "print()",
    "print('Same model, single request, fp16 — context length × n_kv_heads dominates.')",
    "print('GQA-8 vs MHA: 8x cheaper KV cache, lets you batch more requests on the same GPU.')",
))

# 1.3 Visualization
cells.append(code(
    "# Visualize KV cache growth with context length, all three variants",
    "context_lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768])",
    "n_layers, head_dim = 80, 128",
    "",
    "mha = [kv_cache_bytes(n_layers, 64, c, head_dim) / 1e9 for c in context_lengths]",
    "gqa = [kv_cache_bytes(n_layers, 8,  c, head_dim) / 1e9 for c in context_lengths]",
    "mqa = [kv_cache_bytes(n_layers, 1,  c, head_dim) / 1e9 for c in context_lengths]",
    "",
    "fig, ax = plt.subplots(figsize=(8, 5))",
    "ax.plot(context_lengths, mha, 'o-', label='MHA (64 KV heads)', color='#888', lw=2)",
    "ax.plot(context_lengths, gqa, 'o-', label='GQA (8 KV heads, Llama 2-70B)', color='#1f77b4', lw=2)",
    "ax.plot(context_lengths, mqa, 'o-', label='MQA (1 KV head)', color='#d62728', lw=2)",
    "ax.set_xscale('log', base=2)",
    "ax.set_yscale('log')",
    "ax.set_xticks(context_lengths)",
    "ax.set_xticklabels([f'{c//1024}K' for c in context_lengths])",
    "ax.set_xlabel('Context length (tokens)')",
    "ax.set_ylabel('KV cache (GB) per request, fp16')",
    "ax.set_title('KV cache memory — Llama 2-70B equivalent (80 layers, 128 head_dim)')",
    "ax.legend()",
    "ax.grid(True, which='both', alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()",
))

# =========================================================================
# 2. Param counts
# =========================================================================
cells.append(md(
    "## 2. Reproducing parameter counts (Tables 1, 19)",
    "",
    "Llama 2 sizes: 7B / 13B / 34B / 70B. The 34B and 70B use GQA with 8 KV heads. "
    "Because GQA *removes* parameters from the attention block, the FFN is **bumped up by ~1.3×** "
    "to keep the total parameter count roughly the same.",
))

cells.append(code(
    "configs = {",
    "    '7B':  dict(dim=4096, n_heads=32, n_kv_heads=32, n_layers=32, ffn_mult=1.0),  # MHA",
    "    '13B': dict(dim=5120, n_heads=40, n_kv_heads=40, n_layers=40, ffn_mult=1.0),  # MHA",
    "    '34B': dict(dim=8192, n_heads=64, n_kv_heads=8,  n_layers=48, ffn_mult=1.3),  # GQA, FFN bump",
    "    '70B': dict(dim=8192, n_heads=64, n_kv_heads=8,  n_layers=80, ffn_mult=1.3),  # GQA, FFN bump",
    "}",
    "VOCAB = 32_000  # SentencePiece",
    "",
    "def estimate_params(dim, n_heads, n_kv_heads, n_layers, ffn_mult=1.0,",
    "                    vocab_size=VOCAB, multiple_of=256):",
    "    head_dim = dim // n_heads",
    "    # Attention: Q is full-width, K and V are n_kv_heads * head_dim",
    "    attn = (dim * dim) + 2 * (dim * n_kv_heads * head_dim) + (dim * dim)  # WQ + WK + WV + WO",
    "",
    "    # FFN: SwiGLU at (2/3)*4*dim, optionally multiplied by ffn_mult to compensate GQA",
    "    hidden = int(2 / 3 * 4 * dim * ffn_mult)",
    "    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)",
    "    ffn = 3 * dim * hidden",
    "",
    "    norms = 2 * dim",
    "    block = attn + ffn + norms",
    "    embed = vocab_size * dim",
    "    head = vocab_size * dim",
    "    return n_layers * block + embed + dim + head, hidden",
    "",
    "reported = {'7B': 6.74, '13B': 13.0, '34B': 33.7, '70B': 69.0}",
    "print(f\"{'Model':<6}{'GQA?':>8}{'FFN hidden':>14}{'Estimated':>14}{'Reported':>14}\")",
    "print('-' * 56)",
    "for name, cfg in configs.items():",
    "    n, hidden = estimate_params(**cfg)",
    "    gqa_label = 'GQA-8' if cfg['n_kv_heads'] != cfg['n_heads'] else 'MHA'",
    "    print(f'{name:<6}{gqa_label:>8}{hidden:>14,}{n/1e9:>13.2f}B{reported[name]:>13.1f}B')",
))

# =========================================================================
# 3. RLHF math
# =========================================================================
cells.append(md(
    "## 3. The RLHF math",
    "",
    "Llama 2's biggest contribution. Two pieces to walk through:",
    "1. The **reward model loss** (binary ranking with preference-strength margin)",
    "2. The **PPO objective** (clipped surrogate + KL penalty)",
))

cells.append(md(
    "### 3.1 Reward Model — ranking loss with margin",
    "",
    "Standard pairwise ranking (Bradley-Terry / logistic):",
    "$$L = -\\log \\sigma(r_\\theta(x, y_c) - r_\\theta(x, y_r))$$",
    "",
    "Llama 2 adds a **margin term** scaled by preference strength. Bigger margin for 'significantly better,' "
    "smaller for 'negligibly better':",
    "$$L = -\\log \\sigma(r_\\theta(x, y_c) - r_\\theta(x, y_r) - m(r))$$",
))

cells.append(code(
    "def rm_ranking_loss(r_chosen, r_rejected, margin=0.0):",
    "    \"\"\"Llama 2 §3.2.2 reward model loss with optional preference-strength margin.\"\"\"",
    "    return -F.logsigmoid(r_chosen - r_rejected - margin).mean()",
    "",
    "# Toy demonstration",
    "torch.manual_seed(0)",
    "r_chosen = torch.tensor([1.5, 0.8, 2.1, -0.3])",
    "r_rejected = torch.tensor([0.9, 0.7, 1.0, -0.2])",
    "",
    "# Margin from Llama 2 Table 27 (Margin Large variant)",
    "rating_to_margin = {'sig_better': 3.0, 'better': 2.0, 'slightly_better': 1.0, 'negligibly': 0.0}",
    "ratings = ['sig_better', 'sig_better', 'sig_better', 'negligibly']",
    "margins = torch.tensor([rating_to_margin[r] for r in ratings])",
    "",
    "loss_no_margin = rm_ranking_loss(r_chosen, r_rejected, margin=0.0)",
    "loss_with_margin = rm_ranking_loss(r_chosen, r_rejected, margin=margins)",
    "",
    "print(f'Without margin: {loss_no_margin.item():.4f}')",
    "print(f'With margin   : {loss_with_margin.item():.4f}  (penalizes the 3 sig-better pairs harder)')",
    "print()",
    "print('Margin term forces the RM to assign more extreme scores when humans were confident.')",
))

cells.append(md(
    "### 3.2 PPO — clipped surrogate + KL penalty",
    "",
    "The Llama 2 PPO objective (§3.2.3, Eq. 4):",
    "$$R(g | p) = \\tilde{R}_c(g | p) - \\beta \\, D_{KL}(\\pi_\\theta(g|p) \\,\\|\\, \\pi_0(g|p))$$",
    "",
    "where $R_c$ is the piecewise safety/helpfulness reward, $\\pi_0$ is the SFT model (frozen), "
    "and $\\beta$ is the KL coefficient (0.01 for 7B/13B, 0.005 for 34B/70B).",
    "",
    "The PPO clipped surrogate prevents big policy updates:",
    "$$L^{CLIP} = \\mathbb{E}\\Big[\\min(\\rho_t A_t, \\, \\text{clip}(\\rho_t, 1-\\epsilon, 1+\\epsilon) A_t)\\Big],"
    "\\quad \\rho_t = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{old}(a_t|s_t)}$$",
    "",
    "with $\\epsilon = 0.2$.",
))

cells.append(code(
    "def piecewise_reward(r_safety, r_helpful, is_safety_prompt, threshold=0.15):",
    "    \"\"\"Llama 2 §3.2.3 — safety overrides helpfulness when prompt is flagged",
    "    or safety score is below threshold (paper: 0.15).\"\"\"",
    "    use_safety = is_safety_prompt | (r_safety < threshold)",
    "    return torch.where(use_safety, r_safety, r_helpful)",
    "",
    "def ppo_clipped_loss(log_probs_new, log_probs_old, advantages, clip_eps=0.2):",
    "    \"\"\"Standard PPO clipped surrogate.\"\"\"",
    "    rho = torch.exp(log_probs_new - log_probs_old)",
    "    unclipped = rho * advantages",
    "    clipped = torch.clamp(rho, 1 - clip_eps, 1 + clip_eps) * advantages",
    "    return -torch.min(unclipped, clipped).mean()",
    "",
    "# Demo on synthetic data",
    "n = 1000",
    "r_safety = torch.rand(n)  # in [0, 1]",
    "r_helpful = torch.rand(n)",
    "is_safety = torch.rand(n) < 0.3",
    "rewards = piecewise_reward(r_safety, r_helpful, is_safety, threshold=0.15)",
    "",
    "# Visualize: when does safety override?",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))",
    "for ax, label, r in zip(axes, ['Safety', 'Helpful'], [r_safety, r_helpful]):",
    "    ax.scatter(r[~is_safety], rewards[~is_safety], c='#1f77b4', alpha=0.5, s=10, label='non-safety prompt')",
    "    ax.scatter(r[is_safety], rewards[is_safety], c='#d62728', alpha=0.5, s=10, label='safety prompt')",
    "    ax.axvline(0.15, color='k', linestyle='--', alpha=0.5, label='threshold 0.15')",
    "    ax.set_xlabel(f'{label} RM score'); ax.set_ylabel('Final reward used in PPO')",
    "    ax.set_title(f'Final reward vs {label} RM score')",
    "    ax.legend(loc='lower right', fontsize=8)",
    "    ax.grid(alpha=0.3)",
    "plt.tight_layout(); plt.show()",
    "",
    "print('Below 0.15 safety score (or on safety prompts), the safety RM overrides the helpfulness RM.')",
))

# =========================================================================
# 4. Visualizing key results
# =========================================================================
cells.append(md(
    "## 4. Visualizing key results",
    "",
    "Numbers transcribed from the paper's tables and figures. Faithful to source.",
))

# 4.1 Base model comparison
cells.append(md("### 4.1 Llama 2 base model improvement over LLaMA 1 (Tables 3, 19, 21, 22)"))

cells.append(code(
    "benchmarks = ['MMLU\\n(5-shot)', 'BBH\\n(3-shot)', 'GSM8K\\n(8-shot)', 'HumanEval\\npass@1', 'TriviaQA\\n1-shot']",
    "llama1_65 = [63.4, 43.5, 50.9, 23.7, 84.5]",
    "llama2_70 = [68.9, 51.2, 56.8, 29.9, 85.0]",
    "gpt35     = [70.0, np.nan, 57.1, 48.1, np.nan]",
    "gpt4      = [86.4, np.nan, 92.0, 67.0, np.nan]",
    "",
    "x = np.arange(len(benchmarks)); w = 0.2",
    "fig, ax = plt.subplots(figsize=(11, 5))",
    "ax.bar(x - 1.5 * w, llama1_65, w, label='Llama 1-65B', color='#888')",
    "ax.bar(x - 0.5 * w, llama2_70, w, label='Llama 2-70B', color='#1f77b4')",
    "ax.bar(x + 0.5 * w, gpt35, w, label='GPT-3.5', color='#2ca02c')",
    "ax.bar(x + 1.5 * w, gpt4, w, label='GPT-4', color='#d62728')",
    "ax.set_xticks(x); ax.set_xticklabels(benchmarks)",
    "ax.set_ylabel('Score')",
    "ax.set_title('Base-model performance — Llama 2-70B vs Llama 1-65B vs GPT-3.5/4')",
    "ax.legend(); ax.grid(axis='y', alpha=0.3)",
    "plt.tight_layout(); plt.show()",
))

# 4.2 Win rate vs ChatGPT
cells.append(md(
    "### 4.2 Llama 2-Chat human helpfulness win rate (Figure 1)",
    "",
    "Direct human comparison vs ChatGPT and other open chat models, ~4K prompts, 3 raters each.",
))

cells.append(code(
    "# From Figure 1 of the paper — human win/tie/loss rates",
    "matchups = [",
    "    ('Llama2-70B-chat\\nvs ChatGPT-0301',     35.9, 31.5, 32.5),",
    "    ('Llama2-70B-chat\\nvs PaLM-Bison',       53.0, 24.6, 22.4),",
    "    ('Llama2-34B-chat\\nvs Falcon-40B-instr', 76.3, 14.6,  9.1),",
    "    ('Llama2-34B-chat\\nvs Vicuna-33B-v1.3',  37.2, 31.6, 31.2),",
    "    ('Llama2-13B-chat\\nvs Vicuna-13B-v1.1',  45.4, 29.8, 24.9),",
    "    ('Llama2-7B-chat\\nvs MPT-7B-chat',       61.1, 20.9, 18.0),",
    "]",
    "",
    "labels = [m[0] for m in matchups]",
    "wins   = [m[1] for m in matchups]",
    "ties   = [m[2] for m in matchups]",
    "losses = [m[3] for m in matchups]",
    "",
    "fig, ax = plt.subplots(figsize=(10, 5))",
    "y = np.arange(len(labels))",
    "ax.barh(y, wins, color='#1f77b4', label='Win')",
    "ax.barh(y, ties, left=wins, color='#aec7e8', label='Tie')",
    "ax.barh(y, losses, left=[w + t for w, t in zip(wins, ties)], color='#d62728', label='Loss')",
    "for i, (w, t, l) in enumerate(zip(wins, ties, losses)):",
    "    ax.text(w / 2, i, f'{w:.0f}', ha='center', va='center', color='white', fontsize=9)",
    "    ax.text(w + t / 2, i, f'{t:.0f}', ha='center', va='center', fontsize=9)",
    "    ax.text(w + t + l / 2, i, f'{l:.0f}', ha='center', va='center', color='white', fontsize=9)",
    "ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)",
    "ax.set_xlabel('% of human comparisons'); ax.set_xlim(0, 100)",
    "ax.set_title('Llama 2-Chat helpfulness vs other chat models (Figure 1)')",
    "ax.legend(loc='lower right'); ax.grid(axis='x', alpha=0.3)",
    "plt.tight_layout(); plt.show()",
))

# 4.3 Safety violations
cells.append(md("### 4.3 Safety violations (Figure 17)"))

cells.append(code(
    "# Figure 17 — adversarial-prompt violation rate (lower = safer)",
    "models = ['Llama-2\\n7b-chat', 'Llama-2\\n13b-chat', 'Llama-2\\n34b-chat', 'Llama-2\\n70b-chat',",
    "          'MPT\\n7b-chat', 'Vicuna\\n13b-v1.1', 'Vicuna\\n33b-v1.3',",
    "          'Falcon\\n40b-instr', 'PaLM\\nBison', 'ChatGPT\\n0301']",
    "violations = [3, 4, 7, 4, 21, 25, 38, 7.5, 28, 7]   # approx from Figure 17",
    "is_llama2 = [True]*4 + [False]*6",
    "colors = ['#1f77b4' if x else '#aec7e8' for x in is_llama2]",
    "",
    "fig, ax = plt.subplots(figsize=(11, 4.5))",
    "ax.bar(models, violations, color=colors)",
    "for i, v in enumerate(violations):",
    "    ax.text(i, v + 0.7, f'{v}', ha='center', fontsize=9)",
    "ax.set_ylabel('Violation % (lower is safer)')",
    "ax.set_title('Adversarial-prompt safety violations (Figure 17)')",
    "ax.grid(axis='y', alpha=0.3)",
    "plt.xticks(rotation=15, fontsize=9)",
    "plt.tight_layout(); plt.show()",
))

# 4.4 RLHF V1 -> V5
cells.append(md(
    "### 4.4 RLHF V1 → V5 progression (Figure 11)",
    "",
    "Each iteration: collect preference data on the latest checkpoint, train new RMs, run rejection sampling "
    "(V1–V4) or rejection sampling + PPO (V5). Both helpfulness and harmlessness rise across versions.",
))

cells.append(code(
    "# Approximate from Figure 11 (right, GPT-4 judge): win rate of variant vs ChatGPT",
    "stages =     ['SFT-v1', 'SFT-v2', 'RLHF-v1', 'RLHF-v2', 'RLHF-v3', 'RLHF-v4', 'RLHF-v5\\n(no PPO)', 'RLHF-v5\\n(with PPO)']",
    "helpfulness = [12,       40,        45,         52,        58,        65,         68,                    72]",
    "harmlessness= [11,       30,        40,         50,        58,        65,         70,                    75]",
    "",
    "x = np.arange(len(stages))",
    "fig, ax = plt.subplots(figsize=(11, 4.5))",
    "ax.plot(x, helpfulness, 'o-', label='Helpfulness', color='#1f77b4', lw=2)",
    "ax.plot(x, harmlessness, 's-', label='Harmlessness', color='#d62728', lw=2)",
    "ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=9)",
    "ax.set_ylabel('Win rate vs ChatGPT (%, GPT-4 judge)')",
    "ax.set_title('Llama 2-Chat-70B progression across SFT/RLHF stages (approx Figure 11)')",
    "ax.axhline(50, color='k', linestyle='--', alpha=0.5, label='Parity with ChatGPT')",
    "ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 90)",
    "plt.tight_layout(); plt.show()",
    "",
    "print('Iterative RLHF compounds — both axes lift through each version.')",
    "print('Note: PPO is added only at V5; V1-V4 are pure rejection sampling fine-tuning.')",
))

# 4.5 Reward model scaling
cells.append(md("### 4.5 Reward model scaling (Figure 6)"))

cells.append(code(
    "# Approximate from Figure 6 — RM accuracy vs preference-data batch number",
    "batches = np.arange(1, 15)",
    "rm_70b = np.array([0.51, 0.55, 0.55, 0.58, 0.59, 0.60, 0.61, 0.61, 0.62, 0.62, 0.62, 0.625, 0.63, 0.63])",
    "rm_13b = np.array([0.52, 0.55, 0.55, 0.57, 0.58, 0.59, 0.60, 0.59, 0.60, 0.61, 0.60, 0.61, 0.61, 0.61])",
    "rm_7b  = np.array([0.52, 0.54, 0.55, 0.57, 0.57, 0.58, 0.58, 0.58, 0.59, 0.59, 0.60, 0.60, 0.60, 0.61])",
    "",
    "fig, ax = plt.subplots(figsize=(8, 4.5))",
    "ax.plot(batches, rm_7b,  'o-', label='RM 7B',  color='#888')",
    "ax.plot(batches, rm_13b, 's-', label='RM 13B', color='#1f77b4')",
    "ax.plot(batches, rm_70b, '^-', label='RM 70B', color='#d62728')",
    "ax.axhline(0.59, color='k', linestyle='--', alpha=0.5, label='GPT-4 judge')",
    "ax.set_xlabel('Preference-data batch (weekly)')",
    "ax.set_ylabel('RM accuracy on Meta Helpfulness test')",
    "ax.set_title('Reward model scaling — accuracy vs data and model size (Figure 6)')",
    "ax.legend(); ax.grid(alpha=0.3)",
    "plt.tight_layout(); plt.show()",
    "",
    "print('Both data and model size still help; RMs have not saturated at 14 batches of data.')",
))

# =========================================================================
cells.append(md(
    "## 5. Wrap-up",
    "",
    "Two things stick after re-implementing the paper:",
    "",
    "- **GQA is a free lunch at 70B scale** — quality near MHA, KV-cache memory near MQA. The paper's main reusable architectural contribution.",
    "- **The post-training stack is the contribution.** Two RMs, iterative preference collection, rejection sampling (V1–V4) + PPO (V5), GAtt for multi-turn — all of this became the open-model alignment template through 2024.",
    "",
    "**Linked notes:**",
    "- `wiki/sources/llama2-2023.md` — structured paper notes",
    "- `wiki/concepts/gqa.md`, `rlhf.md`, `reward-modeling.md`, `ppo.md`, `rejection-sampling-finetuning.md`, `sft.md`, `ghost-attention.md`",
    "",
    "**Next paper in queue:** Mistral 7B — sliding-window attention + GQA, the small-model successor of LLaMA 1.",
))

# =========================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT}  ({OUT.stat().st_size:,} bytes, {len(cells)} cells)")
