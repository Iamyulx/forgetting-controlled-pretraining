# Continual Pretraining with Forgetting Control

This project simulates **catastrophic forgetting** during **continual pretraining** of a small language model and compares four strategies: 

- `sequential_baseline`: continues pretraining using only new data
- `replay_only`: mixes in old samples from a `replay buffer`.
- `ewc_only`: applies EWC-style regularization.
- `replay_plus_ewc`: combines replay and EWC.

The core idea is not to fine-tune for a downstream task, but to maintain the **same language modeling objective** while the corpus distribution shifts. This makes it a **continual pretraining** experiment.

## What’s Included

- Generation of two synthetic corpora with domain shift:
  - old domain: scientific/technical
  - new domain: financial/regulatory
- Small causal Transformer-based model
- `ReplayBuffer` with reservoir sampling
- Diagonal Fisher estimation for EWC
- Experiment runner with per-epoch metrics
- Export of:
  - `history.csv`
  - `summary.csv`
  - `dataset_preview.json`
  - `.png` plots

## Reported Metrics

- `old_val_loss` and `old_val_perplexity`
- `new_val_loss` and `new_val_perplexity`
- `old_val_accuracy` and `new_val_accuracy`
- `forgetting_score`: how much the old domain degrades after continual pretraining
- `retention_ratio`: fraction of original performance preserved
- `plasticity_gain`: improvement on the new domain during stage 2

## Structure

```text
run_experiment.py
src/forgetting_control/data.py
src/forgetting_control/model.py
src/forgetting_control/strategies.py
src/forgetting_control/experiment.py
```

## How to Run

Quick mode to validate the pipeline:

```bash
python run_experiment.py --quick
```

Full run:

```bash
python run_experiment.py --output-dir outputs/full_run
```

## Expected Outputs

After completion, the output directory will contain:

- `history.csv`: full training history per experiment and epoch
- `summary.csv`: final comparison of strategies
- `validation_curves.png`: old/new perplexity curves
- `strategy_comparison.png`: bar charts for forgetting, retention, and plasticity

## Interpretacion

- If `sequential_baseline` shows a large increase in `old_val_perplexity`, there is clear forgetting.
- If `replay_only` reduces that damage, the buffer is helping retain prior knowledge.
- If `ewc_only` also improves retention, EWC is constraining destructive updates on important weights.
- If `replay_plus_ewc` achieves strong retention without hurting adaptation to the new domain, it provides the strongest evidence for the combined approach.

