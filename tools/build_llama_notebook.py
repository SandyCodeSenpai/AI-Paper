"""Build outputs/llama1-2023-notebook.ipynb from scratch using stdlib json only.

Run: python3 tools/build_llama_notebook.py

The notebook walks through the LLaMA paper (Touvron et al., 2023):
  1. Implements RMSNorm, SwiGLU, RoPE, and a LLaMA decoder block in PyTorch
  2. Reproduces the parameter counts from Table 2
  3. Visualizes headline results (commonsense, toxicity, carbon, scaling)

To run the notebook itself, install: torch, matplotlib, numpy, jupyter.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "llama1-2023-notebook.ipynb"


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split(lines),
    }


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
    "# LLaMA Paper Walkthrough — Architecture & Results",
    "",
    "Companion notebook to **LLaMA: Open and Efficient Foundation Language Models** "
    "(Touvron et al., Meta AI, Feb 2023, [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)).",
    "",
    "Linked notes: `wiki/sources/llama1-2023.md` and concept articles "
    "`wiki/concepts/{rmsnorm, swiglu, rope, pre-normalization, chinchilla-scaling}.md`.",
    "",
    "**What this notebook does:**",
    "1. Implements the three architectural ingredients in PyTorch — RMSNorm, SwiGLU, RoPE",
    "2. Builds a minimal LLaMA-style decoder block",
    "3. Reproduces the parameter counts from Table 2 (7B / 13B / 33B / 65B)",
    "4. Visualizes the headline results from the paper (commonsense, toxicity, carbon, scaling)",
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
cells.append(md(
    "## 1. The Architectural Recipe",
    "",
    "LLaMA is a decoder-only Transformer (Vaswani 2017) with three specific upgrades. "
    "This combination became the *de facto* open-weight recipe — Mistral, Mixtral, Phi, "
    "Qwen, Gemma, DeepSeek, OLMo all use it.",
    "",
    "1. **Pre-normalization with RMSNorm** (from GPT-3 / Zhang & Sennrich 2019)",
    "2. **SwiGLU activation** in the FFN (from PaLM / Shazeer 2020)",
    "3. **Rotary Positional Embeddings — RoPE** (from GPT-NeoX / Su 2021)",
))

# 1.1 RMSNorm
cells.append(md(
    "### 1.1 RMSNorm",
    "",
    "Standard LayerNorm subtracts the mean and divides by the standard deviation:",
    "$$\\text{LayerNorm}(x) = \\gamma \\cdot \\frac{x - \\mu}{\\sigma} + \\beta$$",
    "",
    "RMSNorm drops the mean-centering and the bias term:",
    "$$\\text{RMSNorm}(x) = \\gamma \\cdot \\frac{x}{\\sqrt{\\text{mean}(x^2) + \\varepsilon}}$$",
    "",
    "Cheaper (one fewer reduction) and works equivalently at scale. The intuition is "
    "that re-centering wasn't doing much in modern Transformers — only the rescaling was load-bearing.",
))

cells.append(code(
    "class RMSNorm(nn.Module):",
    "    def __init__(self, dim: int, eps: float = 1e-6):",
    "        super().__init__()",
    "        self.weight = nn.Parameter(torch.ones(dim))",
    "        self.eps = eps",
    "",
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
    "        # x: [..., dim]",
    "        rms_inv = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()",
    "        return self.weight * (x * rms_inv)",
    "",
    "# Quick sanity check: post-norm output has unit RMS per token",
    "x = torch.randn(2, 4, 8)",
    "rn = RMSNorm(8)",
    "y = rn(x)",
    "print(f'Input  RMS per token (mean): {x.pow(2).mean(-1).sqrt().mean():.4f}')",
    "print(f'Output RMS per token (mean): {y.pow(2).mean(-1).sqrt().mean():.4f}  (should be ~1.0)')",
))

# 1.2 Pre-normalization
cells.append(md(
    "### 1.2 Pre-normalization (Pre-LN)",
    "",
    "**Where** the norm is applied inside a block:",
    "",
    "- **Post-LN (original Vaswani 2017):** `x' = LayerNorm(x + Sublayer(x))`",
    "- **Pre-LN (modern):**           `x' = x + Sublayer(LayerNorm(x))`",
    "",
    "Pre-LN keeps the residual stream un-normalized — gradients flow through unimpeded — "
    "which is what makes deep Transformers trainable without aggressive warmup.",
))

cells.append(code(
    "def post_ln_block(x, sublayer, norm):",
    "    return norm(x + sublayer(x))      # original",
    "",
    "def pre_ln_block(x, sublayer, norm):",
    "    return x + sublayer(norm(x))      # LLaMA / GPT-3 / modern",
    "",
    "print('Both block forms keep the same shape; only gradient flow differs.')",
))

# 1.3 SwiGLU
cells.append(md(
    "### 1.3 SwiGLU",
    "",
    "Replaces the FFN's ReLU with a **gated** activation:",
    "$$\\text{FFN}(x) = \\bigl(\\text{Swish}(xW_1) \\odot (xW_3)\\bigr) W_2$$",
    "",
    "where $\\text{Swish}(z) = z \\cdot \\sigma(z)$ (also called SiLU) and $\\odot$ is elementwise multiplication.",
    "",
    "Two input projections (W1 = gate, W3 = value) instead of one. To keep the parameter count "
    "comparable to a vanilla 4d FFN, LLaMA uses hidden dim ≈ **(2/3) · 4d**.",
))

cells.append(code(
    "class SwiGLU(nn.Module):",
    "    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):",
    "        super().__init__()",
    "        if hidden_dim is None:",
    "            hidden_dim = int(2 / 3 * 4 * dim)",
    "            # Round up to a multiple of `multiple_of` (LLaMA convention)",
    "            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)",
    "        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate",
    "        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # value",
    "        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # out",
    "        self.hidden_dim = hidden_dim",
    "",
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
    "        return self.w2(F.silu(self.w1(x)) * self.w3(x))",
    "",
    "# Compare param counts: SwiGLU at (2/3)*4d vs vanilla ReLU FFN at 4d",
    "dim = 4096",
    "ffn = SwiGLU(dim)",
    "swiglu_params = sum(p.numel() for p in ffn.parameters())",
    "vanilla_params = 2 * dim * (4 * dim)  # two linears (in, out) at 4d hidden",
    "print(f'SwiGLU  hidden_dim = {ffn.hidden_dim}')",
    "print(f'SwiGLU  params  = {swiglu_params:>14,}')",
    "print(f'Vanilla params  = {vanilla_params:>14,}  (4d ReLU FFN, for comparison)')",
    "print(f'Ratio           = {swiglu_params / vanilla_params:.3f}')",
))

# 1.4 RoPE
cells.append(md(
    "### 1.4 Rotary Positional Embeddings (RoPE)",
    "",
    "Pair up the dim into 2D blocks. For position $m$, rotate each block by angle $m \\cdot \\theta_i$ where",
    "$$\\theta_i = 10000^{-2(i-1)/d}$$",
    "",
    "Geometric series of frequencies, exactly like sinusoidal embeddings — but applied to "
    "Q and K at every attention layer instead of added to the input.",
    "",
    "**Why this works:** $\\langle q_m, k_n \\rangle$ after rotation depends only on the relative "
    "offset $(m - n)$, not on absolute positions. So attention becomes naturally relative-position-aware "
    "without any extra parameters or biases.",
))

cells.append(code(
    "def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):",
    "    \"\"\"Returns complex-valued rotation factors of shape [max_seq_len, head_dim/2].\"\"\"",
    "    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))",
    "    t = torch.arange(max_seq_len).float()",
    "    freqs = torch.outer(t, freqs)              # [seq_len, head_dim/2]",
    "    return torch.polar(torch.ones_like(freqs), freqs)  # e^{i·m·θ}",
    "",
    "def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):",
    "    \"\"\"xq, xk: [bsz, seq_len, n_heads, head_dim]\"\"\"",
    "    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))",
    "    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))",
    "    fc = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])",
    "    xq_out = torch.view_as_real(xq_ * fc).flatten(-2)",
    "    xk_out = torch.view_as_real(xk_ * fc).flatten(-2)",
    "    return xq_out.type_as(xq), xk_out.type_as(xk)",
    "",
    "# Verify the relative-position property numerically.",
    "# If RoPE works, q@k.T should be invariant under a shared shift in positions.",
    "head_dim, seq_len, n_heads = 64, 16, 1",
    "fc = precompute_rope_freqs(head_dim, seq_len * 2)",
    "",
    "q = torch.randn(1, seq_len, n_heads, head_dim)",
    "k = torch.randn(1, seq_len, n_heads, head_dim)",
    "",
    "# Put q,k at positions [0..seq_len)",
    "q1, k1 = apply_rope(q, k, fc[:seq_len])",
    "# Put q,k at positions [seq_len..2*seq_len)  -- same relative offsets!",
    "q2, k2 = apply_rope(q, k, fc[seq_len:2 * seq_len])",
    "",
    "scores1 = (q1 @ k1.transpose(-2, -1))",
    "scores2 = (q2 @ k2.transpose(-2, -1))",
    "",
    "diff = (scores1 - scores2).abs().max().item()",
    "print(f'Max abs difference between attention scores under shifted positions: {diff:.2e}')",
    "print('(Should be ~1e-5 or smaller — confirms relative-position invariance.)')",
))

# 1.5 LLaMA Block
cells.append(md(
    "### 1.5 Putting it together — a LLaMA decoder block",
    "",
    "All three ingredients in one block: pre-norm + RMSNorm wrapping attention with RoPE, "
    "and pre-norm + RMSNorm wrapping a SwiGLU FFN. This is the building block that gets stacked "
    "32 / 40 / 60 / 80 times for the four model sizes.",
))

cells.append(code(
    "class LLaMAAttention(nn.Module):",
    "    def __init__(self, dim: int, n_heads: int):",
    "        super().__init__()",
    "        self.n_heads = n_heads",
    "        self.head_dim = dim // n_heads",
    "        self.wq = nn.Linear(dim, dim, bias=False)",
    "        self.wk = nn.Linear(dim, dim, bias=False)",
    "        self.wv = nn.Linear(dim, dim, bias=False)",
    "        self.wo = nn.Linear(dim, dim, bias=False)",
    "",
    "    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:",
    "        bsz, seq_len, _ = x.shape",
    "        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)",
    "        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim)",
    "        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim)",
    "        q, k = apply_rope(q, k, freqs_cis)",
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
    "",
    "class LLaMABlock(nn.Module):",
    "    def __init__(self, dim: int, n_heads: int):",
    "        super().__init__()",
    "        self.attn_norm = RMSNorm(dim)",
    "        self.ffn_norm = RMSNorm(dim)",
    "        self.attn = LLaMAAttention(dim, n_heads)",
    "        self.ffn = SwiGLU(dim)",
    "",
    "    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:",
    "        x = x + self.attn(self.attn_norm(x), freqs_cis)   # pre-norm residual",
    "        x = x + self.ffn(self.ffn_norm(x))                 # pre-norm residual",
    "        return x",
    "",
    "",
    "# Smoke test",
    "dim, n_heads, seq_len, bsz = 256, 4, 32, 2",
    "block = LLaMABlock(dim, n_heads)",
    "fc = precompute_rope_freqs(dim // n_heads, seq_len)",
    "x = torch.randn(bsz, seq_len, dim)",
    "y = block(x, fc)",
    "print(f'Forward: in {tuple(x.shape)}  ->  out {tuple(y.shape)}')",
    "print(f'Block params (dim={dim}, n_heads={n_heads}): {sum(p.numel() for p in block.parameters()):,}')",
))

# =========================================================================
cells.append(md(
    "## 2. Reproducing Table 2 — parameter counts",
    "",
    "Table 2 of the paper lists 6.7B / 13.0B / 32.5B / 65.2B parameters for the four sizes. "
    "Let's reconstruct those numbers from architecture alone.",
    "",
    "Each transformer block has:",
    "- Attention: 4 × (dim × dim) for Q, K, V, O projections",
    "- FFN (SwiGLU): 3 × (dim × hidden_dim) for W1, W2, W3",
    "- 2 × dim for the two RMSNorm scales",
    "",
    "Plus token embedding + final norm + output projection (untied in LLaMA).",
))

cells.append(code(
    "configs = {",
    "    '7B':  dict(dim=4096, n_heads=32, n_layers=32),",
    "    '13B': dict(dim=5120, n_heads=40, n_layers=40),",
    "    '33B': dict(dim=6656, n_heads=52, n_layers=60),",
    "    '65B': dict(dim=8192, n_heads=64, n_layers=80),",
    "}",
    "VOCAB = 32_000  # LLaMA SentencePiece",
    "",
    "def estimate_params(dim, n_heads, n_layers, vocab_size=VOCAB, multiple_of=256):",
    "    hidden = int(2 / 3 * 4 * dim)",
    "    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)",
    "    attn = 4 * dim * dim                # wq, wk, wv, wo",
    "    ffn = 3 * dim * hidden              # w1, w2, w3",
    "    norms = 2 * dim                     # attn_norm + ffn_norm",
    "    block = attn + ffn + norms",
    "    embed = vocab_size * dim",
    "    head = vocab_size * dim             # untied output",
    "    return n_layers * block + embed + dim + head",
    "",
    "reported = {'7B': 6.7, '13B': 13.0, '33B': 32.5, '65B': 65.2}",
    "print(f\"{'Model':<6}{'Estimated':>14}{'Reported (Table 2)':>22}\")",
    "print('-' * 42)",
    "for name, cfg in configs.items():",
    "    n = estimate_params(**cfg)",
    "    print(f'{name:<6}{n/1e9:>13.2f}B{reported[name]:>20.1f}B')",
))

# =========================================================================
cells.append(md(
    "## 3. Visualizing key results",
    "",
    "Numbers transcribed from the paper's tables. All faithful to source — no fabrication.",
))

# 3.1 Common-sense reasoning
cells.append(md("### 3.1 Common-sense reasoning (Table 3, zero-shot)"))

cells.append(code(
    "# Table 3 — '-' in paper rendered as np.nan",
    "benchmarks = ['BoolQ', 'PIQA', 'SIQA', 'HellaSwag', 'WinoGrande', 'ARC-e', 'ARC-c', 'OBQA']",
    "data = {",
    "    'GPT-3 (175B)':       [60.5, 81.0, np.nan, 78.9, 70.2, 68.8, 51.4, 57.6],",
    "    'Chinchilla (70B)':   [83.7, 81.8, 51.3,   80.8, 74.9, np.nan, np.nan, np.nan],",
    "    'PaLM (540B)':        [88.0, 82.3, np.nan, 83.4, 81.1, 76.6, 53.0, 53.4],",
    "    'LLaMA-13B':          [78.1, 80.1, 50.4,   79.2, 73.0, 74.8, 52.7, 56.4],",
    "    'LLaMA-65B':          [85.3, 82.8, 52.3,   84.2, 77.0, 78.9, 56.0, 60.2],",
    "}",
    "",
    "fig, ax = plt.subplots(figsize=(12, 5))",
    "x = np.arange(len(benchmarks))",
    "width = 0.16",
    "colors = ['#888', '#666', '#444', '#1f77b4', '#d62728']",
    "for i, (label, scores) in enumerate(data.items()):",
    "    ax.bar(x + (i - 2) * width, scores, width, label=label, color=colors[i])",
    "ax.set_xticks(x)",
    "ax.set_xticklabels(benchmarks, rotation=20)",
    "ax.set_ylabel('Zero-shot accuracy (%)')",
    "ax.set_title('Common-sense reasoning — LLaMA-65B vs much larger closed models (Table 3)')",
    "ax.legend(loc='lower right', fontsize=9)",
    "ax.set_ylim(40, 95)",
    "ax.grid(axis='y', alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()",
))

# 3.2 efficiency
cells.append(md(
    "### 3.2 Efficiency — performance per parameter",
    "",
    "LLaMA-13B on most benchmarks beats GPT-3 175B at **13× fewer parameters**.",
))

cells.append(code(
    "# MMLU 5-shot accuracy vs parameter count (Table 9)",
    "models = [",
    "    ('GPT-NeoX',  20,  33.6),",
    "    ('GPT-3',     175, 43.9),",
    "    ('Gopher',    280, 60.0),",
    "    ('Chinchilla', 70, 67.5),",
    "    ('PaLM',      540, 69.3),",
    "    ('LLaMA-7B',    7, 35.1),",
    "    ('LLaMA-13B',  13, 46.9),",
    "    ('LLaMA-33B',  33, 57.8),",
    "    ('LLaMA-65B',  65, 63.4),",
    "]",
    "",
    "fig, ax = plt.subplots(figsize=(9, 5.5))",
    "for name, params, mmlu in models:",
    "    color = '#d62728' if name.startswith('LLaMA') else '#1f77b4'",
    "    ax.scatter(params, mmlu, s=110, color=color, edgecolor='k', zorder=3)",
    "    ax.annotate(name, (params, mmlu), xytext=(7, 5), textcoords='offset points', fontsize=9)",
    "ax.set_xscale('log')",
    "ax.set_xlabel('Parameters (B)')",
    "ax.set_ylabel('MMLU 5-shot accuracy (%)')",
    "ax.set_title('MMLU vs model size — LLaMA family in red')",
    "ax.grid(True, which='both', alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()",
))

# 3.3 toxicity
cells.append(md(
    "### 3.3 Toxicity scales with model size (Table 11)",
    "",
    "An uncomfortable finding: bigger LLaMA models are *more* toxic on RealToxicityPrompts. "
    "And 'respectful' framing helps small models but **hurts** the 65B (0.087 → 0.141).",
))

cells.append(code(
    "sizes = ['7B', '13B', '33B', '65B']",
    "basic       = [0.106, 0.104, 0.107, 0.128]",
    "respectful  = [0.081, 0.095, 0.087, 0.141]",
    "",
    "fig, ax = plt.subplots(figsize=(7.5, 4.5))",
    "x = np.arange(len(sizes))",
    "ax.bar(x - 0.2, basic, 0.4, label='Basic prompt', color='#666')",
    "ax.bar(x + 0.2, respectful, 0.4, label='\"Respectful\" prompt', color='#1f77b4')",
    "ax.set_xticks(x); ax.set_xticklabels(sizes)",
    "ax.set_ylabel('Mean toxicity score')",
    "ax.set_title('RealToxicityPrompts — toxicity rises with scale (Table 11)')",
    "ax.legend()",
    "ax.grid(axis='y', alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()",
))

# 3.4 carbon
cells.append(md("### 3.4 Carbon footprint (Table 15)"))

cells.append(code(
    "models = ['LLaMA-7B', 'LLaMA-13B', 'LLaMA-33B', 'LLaMA-65B', 'OPT-175B', 'BLOOM-175B']",
    "co2 = [14, 23, 90, 173, 137, 183]  # tCO2eq",
    "colors = ['#d62728'] * 4 + ['#888'] * 2",
    "",
    "fig, ax = plt.subplots(figsize=(8.5, 4.5))",
    "ax.bar(models, co2, color=colors)",
    "ax.set_ylabel('tCO$_2$eq')",
    "ax.set_title('Training carbon footprint — LLaMA family vs other large open models')",
    "for i, v in enumerate(co2):",
    "    ax.text(i, v + 3, str(v), ha='center', fontsize=10)",
    "plt.xticks(rotation=15)",
    "ax.grid(axis='y', alpha=0.3)",
    "plt.tight_layout()",
    "plt.show()",
))

# =========================================================================
cells.append(md(
    "## 4. Chinchilla vs LLaMA — tokens per parameter",
    "",
    "Chinchilla (Hoffmann 2022) said: for compute-optimal training, use ~20 tokens per parameter. "
    "LLaMA pushed *past* this — even when training is no longer 'compute-optimal,' the resulting "
    "smaller model is much cheaper to **serve**. This reframing — inference-optimal vs. compute-optimal — "
    "is the conceptual move that made the open-LLM ecosystem possible.",
))

cells.append(code(
    "models = ['LLaMA-7B', 'LLaMA-13B', 'LLaMA-33B', 'LLaMA-65B']",
    "params_b = [7, 13, 33, 65]",
    "tokens_b = [1000, 1000, 1400, 1400]",
    "tokens_per_param = [t / p for t, p in zip(tokens_b, params_b)]",
    "",
    "print(f\"{'Model':<12}{'Params':>10}{'Tokens':>10}{'Tokens/Param':>16}\")",
    "print('-' * 48)",
    "for m, p, t, tpp in zip(models, params_b, tokens_b, tokens_per_param):",
    "    print(f'{m:<12}{p:>9}B{t:>9}B{tpp:>15.1f}')",
    "",
    "print()",
    "print('Chinchilla compute-optimal rule of thumb: ~20 tokens / parameter')",
    "print(f'LLaMA-7B trains at {tokens_per_param[0]:.0f}× over Chinchilla-optimal — by design.')",
    "",
    "fig, ax = plt.subplots(figsize=(8, 4.5))",
    "ax.bar(models, tokens_per_param, color='#1f77b4')",
    "ax.axhline(20, color='#d62728', linestyle='--', label='Chinchilla optimum (~20)')",
    "ax.set_ylabel('Tokens per parameter')",
    "ax.set_title('LLaMA training duration vs Chinchilla compute-optimal point')",
    "ax.legend()",
    "ax.grid(axis='y', alpha=0.3)",
    "for i, v in enumerate(tokens_per_param):",
    "    ax.text(i, v + 3, f'{v:.0f}', ha='center', fontsize=10)",
    "plt.tight_layout()",
    "plt.show()",
))

# =========================================================================
cells.append(md(
    "## 5. Wrap-up",
    "",
    "Key takeaways from re-implementing the paper:",
    "",
    "- The architecture is *simple* — pre-norm + RMSNorm + SwiGLU + RoPE on top of a vanilla decoder Transformer. "
    "  Most of the implementation is ~80 lines of PyTorch.",
    "- The big idea is in **Section 2.4 (efficient implementation)** and the **token-count decision** — both invisible at the model definition level.",
    "- The MMLU gap vs. Chinchilla / PaLM is real, and the paper is honest about it (data composition, not architecture).",
    "- The toxicity-rises-with-scale finding is a quiet warning that subsequent papers (Llama 2 onward) had to address with safety training.",
    "",
    "**Linked notes:**",
    "- `wiki/sources/llama1-2023.md` — structured paper notes",
    "- `wiki/concepts/rmsnorm.md`, `swiglu.md`, `rope.md`, `pre-normalization.md`, `chinchilla-scaling.md`",
    "",
    "**Next paper in queue:** Llama 2 (post-training, RLHF, chat optimization) — same architecture, different focus.",
))

# =========================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Wrote {OUT}  ({OUT.stat().st_size:,} bytes, {len(cells)} cells)")
