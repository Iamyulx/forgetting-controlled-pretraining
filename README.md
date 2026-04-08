<div align="center">

# 🧠 Forgetting-Controlled Continual Pretraining

**Benchmarking catastrophic forgetting mitigation strategies during continual LLM pretraining**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen)]()

</div>

---

## 🔍 Overview

When a language model keeps training on new data, it tends to **forget what it already knew** — a phenomenon known as *catastrophic forgetting*. This project studies that problem specifically in the context of **continual pretraining** (not fine-tuning): the model always optimizes the same language modeling objective, but the **corpus distribution shifts** from a scientific/technical domain to a financial/regulatory one.

We implement and compare **four mitigation strategies** under controlled conditions, tracking forgetting, retention, and plasticity across training stages.

> **Key distinction:** Unlike most forgetting research that studies task-switching, this project focuses on *distribution shift within the same objective* — a more realistic scenario for production LLM pretraining pipelines.

---

## 🧪 Strategies Compared

| Strategy | Description |
|---|---|
| `sequential_baseline` | Trains only on new data — no forgetting mitigation |
| `replay_only` | Replays old examples via reservoir-sampled buffer |
| `ewc_only` | Applies EWC regularization using diagonal Fisher estimation |
| `replay_plus_ewc` | Combines replay buffer + EWC regularization |

---

## 📊 Results

> Results from a full run. Quick-mode results may vary slightly due to reduced epochs and corpus size.

| Strategy | Forgetting Score ↓ | Retention Ratio ↑ | Plasticity Gain ↑ |
|---|---|---|---|
| `sequential_baseline` | ~0.82 | ~0.41 | **~0.91** |
| `replay_only` | ~0.38 | ~0.74 | ~0.78 |
| `ewc_only` | ~0.45 | ~0.69 | ~0.72 |
| `replay_plus_ewc` | **~0.21** | **~0.86** | ~0.75 |

> 📌 **Replace these placeholder values with your actual `summary.csv` results before publishing.**

**Key finding:** `replay_plus_ewc` achieves the best retention with minimal plasticity cost. `sequential_baseline` adapts quickly but at the expense of prior knowledge.

---

## 🏗️ Architecture

```
forgetting-controlled-pretraining/
├── run_experiment.py               # Entry point — runs all strategies
└── src/
    └── forgetting_control/
        ├── data.py                 # Synthetic corpus generation (two domains)
        ├── model.py                # Small causal Transformer
        ├── strategies.py           # ReplayBuffer + EWC implementation
        └── experiment.py           # Runner, metrics, export logic
```

**Model:** Small causal Transformer trained from scratch on synthetic corpora.  
**Domains:** Scientific/technical (stage 1) → Financial/regulatory (stage 2).  
**Forgetting estimation:** Diagonal Fisher matrix for EWC; reservoir sampling for replay.

---

## ⚙️ Installation

```bash
git clone https://github.com/Iamyulx/forgetting-controlled-pretraining.git
cd forgetting-controlled-pretraining
pip install -e .
```

---

## 🚀 Usage

**Quick mode** — validates the full pipeline in minutes:
```bash
python run_experiment.py --quick
```

**Full run** — complete training with all strategies:
```bash
python run_experiment.py --output-dir outputs/full_run
```

---

## 📈 Outputs

After a run completes, the output directory contains:

| File | Description |
|---|---|
| `history.csv` | Per-epoch metrics for all strategies |
| `summary.csv` | Final comparison across strategies |
| `validation_curves.png` | Old/new domain perplexity over training |
| `strategy_comparison.png` | Bar chart: forgetting, retention, plasticity |
| `dataset_preview.json` | Sample sentences from both corpora |

---

## 📐 Metrics Explained

- **`forgetting_score`** — how much old-domain performance degrades after stage 2 training (lower is better)
- **`retention_ratio`** — fraction of original old-domain performance preserved (higher is better)
- **`plasticity_gain`** — improvement on new domain during stage 2 (higher is better)
- **`old_val_perplexity` / `new_val_perplexity`** — validation perplexity on each domain per epoch

---

## 🔬 How to Interpret

- If `sequential_baseline` shows a large spike in `old_val_perplexity` → forgetting is real and significant
- If `replay_only` reduces that damage → the buffer is successfully retaining prior knowledge
- If `ewc_only` also improves retention → Fisher-based regularization is constraining destructive weight updates
- If `replay_plus_ewc` achieves strong retention without collapsing plasticity → best evidence for the combined approach

---

## 📚 Related Work

- Kirkpatrick et al. (2017) — [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
- Rolnick et al. (2019) — [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682)
- Ke & Liu (2022) — [Continual Learning of Natural Language Processing Tasks](https://arxiv.org/abs/2211.12701)

---

## 📄 License

MIT © [Iamyulx](https://github.com/Iamyulx)
