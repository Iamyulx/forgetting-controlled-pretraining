from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from forgetting_control.data import ContinualDatasetBundle, SequenceDataset, build_continual_datasets
from forgetting_control.model import TinyCausalLM
from forgetting_control.strategies import (
    EWCRegularizer,
    ReplayBuffer,
    build_replay_dataset,
    replay_examples_needed,
)


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    replay_ratio: float
    ewc_lambda: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    seq_len: int = 18
    train_sentences_per_domain: int = 1400
    val_sentences_per_domain: int = 280
    batch_size: int = 32
    stage1_epochs: int = 5
    stage2_epochs: int = 7
    buffer_size: int = 320
    lr: float = 3e-3
    weight_decay: float = 1e-4
    fisher_batches: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    output_dir: Path = Path("outputs/latest")
    device: str = "auto"


DEFAULT_STRATEGIES = [
    StrategyConfig(name="sequential_baseline", replay_ratio=0.0, ewc_lambda=0.0),
    StrategyConfig(name="replay_only", replay_ratio=0.35, ewc_lambda=0.0),
    StrategyConfig(name="ewc_only", replay_ratio=0.0, ewc_lambda=8.0),
    StrategyConfig(name="replay_plus_ewc", replay_ratio=0.35, ewc_lambda=8.0),
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config_from_args(args)
    run_experiments(config=config, strategies=DEFAULT_STRATEGIES)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continual pretraining with replay and EWC.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seq-len", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=7)
    parser.add_argument("--train-sentences", type=int, default=1400)
    parser.add_argument("--val-sentences", type=int, default=280)
    parser.add_argument("--buffer-size", type=int, default=320)
    parser.add_argument("--fisher-batches", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true", help="Reduce dataset y epocas para smoke test.")
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.quick:
        return ExperimentConfig(
            seed=args.seed,
            seq_len=args.seq_len,
            train_sentences_per_domain=500,
            val_sentences_per_domain=120,
            batch_size=args.batch_size,
            stage1_epochs=2,
            stage2_epochs=3,
            buffer_size=min(args.buffer_size, 160),
            lr=args.lr,
            weight_decay=args.weight_decay,
            fisher_batches=min(args.fisher_batches, 8),
            d_model=min(args.d_model, 48),
            n_heads=args.n_heads,
            n_layers=min(args.n_layers, 1),
            dropout=args.dropout,
            output_dir=args.output_dir,
            device=args.device,
        )

    return ExperimentConfig(
        seed=args.seed,
        seq_len=args.seq_len,
        train_sentences_per_domain=args.train_sentences,
        val_sentences_per_domain=args.val_sentences,
        batch_size=args.batch_size,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        buffer_size=args.buffer_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fisher_batches=args.fisher_batches,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        output_dir=args.output_dir,
        device=args.device,
    )


def run_experiments(config: ExperimentConfig, strategies: list[StrategyConfig]) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(resolve_device(config.device))

    set_seed(config.seed)
    datasets = build_continual_datasets(
        seq_len=config.seq_len,
        train_sentences_per_domain=config.train_sentences_per_domain,
        val_sentences_per_domain=config.val_sentences_per_domain,
        seed=config.seed,
    )

    _write_dataset_preview(output_dir, datasets, config, strategies, str(device))

    history_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for experiment_index, strategy in enumerate(strategies):
        experiment_seed = config.seed + (experiment_index * 101)
        set_seed(experiment_seed)

        model = TinyCausalLM(
            vocab_size=datasets.vocab_size,
            max_seq_len=config.seq_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        old_train_loader = make_dataloader(
            datasets.old_train,
            batch_size=config.batch_size,
            shuffle=True,
            seed=experiment_seed,
        )

        for epoch in range(1, config.stage1_epochs + 1):
            train_stats = train_one_epoch(model, optimizer, old_train_loader, device=device, ewc=None)
            old_eval = evaluate_model(model, datasets.old_val, config.batch_size, device)
            new_eval = evaluate_model(model, datasets.new_val, config.batch_size, device)
            history_rows.append(
                _build_history_row(
                    strategy_name=strategy.name,
                    phase="stage1_pretraining",
                    epoch=epoch,
                    timeline_epoch=epoch,
                    train_stats=train_stats,
                    old_eval=old_eval,
                    new_eval=new_eval,
                    forgetting_score=0.0,
                    retention_ratio=1.0,
                    plasticity_gain=0.0,
                )
            )

        stage1_rows = [
            row for row in history_rows if row["experiment"] == strategy.name and row["phase"] == "stage1_pretraining"
        ]
        stage1_reference_loss = min(float(row["old_val_loss"]) for row in stage1_rows)
        stage1_reference_ppl = min(float(row["old_val_perplexity"]) for row in stage1_rows)
        pre_stage2_new_eval = evaluate_model(model, datasets.new_val, config.batch_size, device)

        replay_buffer = ReplayBuffer(max_examples=config.buffer_size, seed=experiment_seed)
        replay_buffer.add_examples(datasets.old_train.examples)
        stage2_dataset = build_replay_dataset(
            new_dataset=datasets.new_train,
            replay_buffer=replay_buffer,
            replay_ratio=strategy.replay_ratio,
            seed=experiment_seed + 17,
        )
        stage2_loader = make_dataloader(
            stage2_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            seed=experiment_seed + 1,
        )

        ewc = None
        if strategy.ewc_lambda > 0:
            fisher_loader = make_dataloader(
                datasets.old_train,
                batch_size=config.batch_size,
                shuffle=True,
                seed=experiment_seed + 2,
            )
            ewc = EWCRegularizer.estimate(
                model=model,
                dataloader=fisher_loader,
                device=device,
                strength=strategy.ewc_lambda,
                max_batches=config.fisher_batches,
            )

        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        for epoch in range(1, config.stage2_epochs + 1):
            train_stats = train_one_epoch(model, optimizer, stage2_loader, device=device, ewc=ewc)
            old_eval = evaluate_model(model, datasets.old_val, config.batch_size, device)
            new_eval = evaluate_model(model, datasets.new_val, config.batch_size, device)
            forgetting_score = old_eval["loss"] - stage1_reference_loss
            retention_ratio = stage1_reference_ppl / max(old_eval["perplexity"], 1e-8)
            plasticity_gain = pre_stage2_new_eval["loss"] - new_eval["loss"]
            history_rows.append(
                _build_history_row(
                    strategy_name=strategy.name,
                    phase="stage2_continual_pretraining",
                    epoch=epoch,
                    timeline_epoch=config.stage1_epochs + epoch,
                    train_stats=train_stats,
                    old_eval=old_eval,
                    new_eval=new_eval,
                    forgetting_score=forgetting_score,
                    retention_ratio=retention_ratio,
                    plasticity_gain=plasticity_gain,
                )
            )

        experiment_rows = [row for row in history_rows if row["experiment"] == strategy.name]
        final_row = experiment_rows[-1]
        summary_rows.append(
            {
                "experiment": strategy.name,
                "replay_ratio": strategy.replay_ratio,
                "ewc_lambda": strategy.ewc_lambda,
                "replay_examples": replay_examples_needed(len(datasets.new_train), strategy.replay_ratio),
                "buffer_size": config.buffer_size,
                "old_val_loss_best_stage1": stage1_reference_loss,
                "old_val_perplexity_best_stage1": stage1_reference_ppl,
                "old_val_loss_final": final_row["old_val_loss"],
                "old_val_perplexity_final": final_row["old_val_perplexity"],
                "old_val_accuracy_final": final_row["old_val_accuracy"],
                "new_val_loss_before_stage2": pre_stage2_new_eval["loss"],
                "new_val_loss_final": final_row["new_val_loss"],
                "new_val_perplexity_final": final_row["new_val_perplexity"],
                "new_val_accuracy_final": final_row["new_val_accuracy"],
                "forgetting_score": final_row["forgetting_score"],
                "retention_ratio": final_row["retention_ratio"],
                "plasticity_gain": final_row["plasticity_gain"],
            }
        )

    history_df = pd.DataFrame(history_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("forgetting_score")

    history_df.to_csv(output_dir / "history.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    plot_validation_curves(history_df, output_dir, config.stage1_epochs)
    plot_strategy_comparison(summary_df, output_dir)

    print("\nResumen final:")
    print(
        summary_df[
            [
                "experiment",
                "forgetting_score",
                "retention_ratio",
                "plasticity_gain",
                "old_val_perplexity_final",
                "new_val_perplexity_final",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:0.4f}")
    )
    print(f"\nArtifacts written to: {output_dir.resolve()}")
    return history_df, summary_df


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def make_dataloader(
    dataset: SequenceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_one_epoch(
    model: TinyCausalLM,
    optimizer: AdamW,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    ewc: EWCRegularizer | None,
) -> dict[str, float]:
    model.train()
    total_lm_loss = 0.0
    total_penalty = 0.0
    total_objective = 0.0
    steps = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        penalty = ewc.penalty(model) if ewc is not None else torch.zeros((), device=device)
        objective = lm_loss + penalty
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_lm_loss += float(lm_loss.detach())
        total_penalty += float(penalty.detach())
        total_objective += float(objective.detach())
        steps += 1

    return {
        "train_lm_loss": total_lm_loss / max(steps, 1),
        "train_penalty": total_penalty / max(steps, 1),
        "train_objective": total_objective / max(steps, 1),
    }


@torch.no_grad()
def evaluate_model(
    model: TinyCausalLM,
    dataset: SequenceDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dataloader = make_dataloader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction="sum",
        )
        predictions = logits.argmax(dim=-1)
        total_correct += int((predictions == labels).sum().item())
        total_loss += float(loss.item())
        total_tokens += int(labels.numel())

    mean_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20.0))
    accuracy = total_correct / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
    }


def _build_history_row(
    strategy_name: str,
    phase: str,
    epoch: int,
    timeline_epoch: int,
    train_stats: dict[str, float],
    old_eval: dict[str, float],
    new_eval: dict[str, float],
    forgetting_score: float,
    retention_ratio: float,
    plasticity_gain: float,
) -> dict[str, float | int | str]:
    return {
        "experiment": strategy_name,
        "phase": phase,
        "epoch": epoch,
        "timeline_epoch": timeline_epoch,
        "train_lm_loss": train_stats["train_lm_loss"],
        "train_penalty": train_stats["train_penalty"],
        "train_objective": train_stats["train_objective"],
        "old_val_loss": old_eval["loss"],
        "old_val_perplexity": old_eval["perplexity"],
        "old_val_accuracy": old_eval["accuracy"],
        "new_val_loss": new_eval["loss"],
        "new_val_perplexity": new_eval["perplexity"],
        "new_val_accuracy": new_eval["accuracy"],
        "forgetting_score": forgetting_score,
        "retention_ratio": retention_ratio,
        "plasticity_gain": plasticity_gain,
    }


def _write_dataset_preview(
    output_dir: Path,
    datasets: ContinualDatasetBundle,
    config: ExperimentConfig,
    strategies: list[StrategyConfig],
    device: str,
) -> None:
    payload = {
        "config": {
            **{key: value for key, value in asdict(config).items() if key != "output_dir"},
            "output_dir": str(config.output_dir),
            "device_resolved": device,
        },
        "strategies": [asdict(strategy) for strategy in strategies],
        "vocab_size": datasets.vocab_size,
        "old_train_examples": len(datasets.old_train),
        "new_train_examples": len(datasets.new_train),
        "sample_sentences": datasets.sample_sentences,
    }
    with (output_dir / "dataset_preview.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_validation_curves(history_df: pd.DataFrame, output_dir: Path, stage1_epochs: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for experiment, group in history_df.groupby("experiment"):
        ordered = group.sort_values("timeline_epoch")
        axes[0].plot(
            ordered["timeline_epoch"],
            ordered["old_val_perplexity"],
            marker="o",
            linewidth=2,
            label=experiment,
        )
        axes[1].plot(
            ordered["timeline_epoch"],
            ordered["new_val_perplexity"],
            marker="o",
            linewidth=2,
            label=experiment,
        )

    axes[0].set_title("Old-domain perplexity")
    axes[1].set_title("New-domain perplexity")
    for axis in axes:
        axis.axvline(stage1_epochs, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Timeline epoch")
        axis.set_ylabel("Perplexity")
        axis.grid(alpha=0.3)
        axis.legend()

    fig.savefig(output_dir / "validation_curves.png", dpi=180)
    plt.close(fig)


def plot_strategy_comparison(summary_df: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary_df.sort_values("forgetting_score")
    positions = range(len(ordered))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].bar(positions, ordered["forgetting_score"], color="#c0392b")
    axes[0].set_title("Forgetting score")
    axes[0].set_ylabel("Old loss delta")

    axes[1].bar(positions, ordered["retention_ratio"], color="#2980b9")
    axes[1].set_title("Retention ratio")
    axes[1].set_ylabel("Best old PPL / final old PPL")

    axes[2].bar(positions, ordered["plasticity_gain"], color="#27ae60")
    axes[2].set_title("Plasticity gain")
    axes[2].set_ylabel("New loss improvement")

    for axis in axes:
        axis.set_xticks(list(positions))
        axis.set_xticklabels(ordered["experiment"], rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)

    fig.savefig(output_dir / "strategy_comparison.png", dpi=180)
    plt.close(fig)
