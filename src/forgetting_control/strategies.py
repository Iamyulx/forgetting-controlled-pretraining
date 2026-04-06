from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from forgetting_control.data import SequenceDataset


def replay_examples_needed(num_new_examples: int, replay_ratio: float) -> int:
    if replay_ratio <= 0:
        return 0
    if replay_ratio >= 1:
        raise ValueError("replay_ratio must be < 1.")
    return math.ceil((num_new_examples * replay_ratio) / (1 - replay_ratio))


@dataclass
class ReplayBuffer:
    max_examples: int
    seed: int = 0
    examples: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    seen_examples: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def __len__(self) -> int:
        return len(self.examples)

    def add_examples(self, examples: list[tuple[Tensor, Tensor]]) -> None:
        for inputs, labels in examples:
            candidate = (inputs.clone(), labels.clone())
            self.seen_examples += 1
            if len(self.examples) < self.max_examples:
                self.examples.append(candidate)
                continue

            slot = self.rng.randrange(self.seen_examples)
            if slot < self.max_examples:
                self.examples[slot] = candidate

    def sample_examples(self, num_examples: int) -> list[tuple[Tensor, Tensor]]:
        if not self.examples or num_examples <= 0:
            return []
        return [
            tuple(item.clone() for item in self.rng.choice(self.examples))
            for _ in range(num_examples)
        ]


def build_replay_dataset(
    new_dataset: SequenceDataset,
    replay_buffer: ReplayBuffer,
    replay_ratio: float,
    seed: int,
) -> SequenceDataset:
    mixed_examples = new_dataset.clone_examples()
    replay_count = replay_examples_needed(len(new_dataset), replay_ratio)
    mixed_examples.extend(replay_buffer.sample_examples(replay_count))
    random.Random(seed).shuffle(mixed_examples)
    return SequenceDataset.from_examples(
        examples=mixed_examples,
        domain_name=f"{new_dataset.domain_name}_with_replay",
    )


class EWCRegularizer:
    def __init__(
        self,
        reference_params: dict[str, Tensor],
        fisher_diagonal: dict[str, Tensor],
        strength: float,
    ) -> None:
        self.reference_params = reference_params
        self.fisher_diagonal = fisher_diagonal
        self.strength = strength

    @classmethod
    def estimate(
        cls,
        model: nn.Module,
        dataloader: DataLoader[tuple[Tensor, Tensor]],
        device: torch.device,
        strength: float,
        max_batches: int,
    ) -> "EWCRegularizer":
        fisher = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        reference_params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        model.eval()
        batch_count = 0
        for inputs, labels in dataloader:
            if batch_count >= max_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)

            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None and name in fisher:
                    fisher[name] += param.grad.detach().pow(2)
            batch_count += 1

        if batch_count == 0:
            raise ValueError("EWC estimation received zero batches.")

        for name in fisher:
            fisher[name] /= batch_count

        model.zero_grad(set_to_none=True)
        model.train()
        return cls(reference_params=reference_params, fisher_diagonal=fisher, strength=strength)

    def penalty(self, model: nn.Module) -> Tensor:
        penalty = torch.zeros((), device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name not in self.fisher_diagonal:
                continue
            penalty = penalty + (
                self.fisher_diagonal[name] * (param - self.reference_params[name]).pow(2)
            ).sum()
        return 0.5 * self.strength * penalty
