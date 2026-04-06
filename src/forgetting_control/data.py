from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,;]")


@dataclass(frozen=True)
class DomainSpec:
    name: str
    subjects: tuple[str, ...]
    verbs: tuple[str, ...]
    objects: tuple[str, ...]
    descriptors: tuple[str, ...]
    contexts: tuple[str, ...]
    rare_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ContinualDatasetBundle:
    old_train: "SequenceDataset"
    old_val: "SequenceDataset"
    new_train: "SequenceDataset"
    new_val: "SequenceDataset"
    vocab: "Vocab"
    old_name: str
    new_name: str
    sample_sentences: dict[str, list[str]]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class Vocab:
    def __init__(self, tokens: Iterable[str]) -> None:
        specials = ["<pad>", "<bos>", "<eos>"]
        ordered_tokens = specials + sorted(set(tokens))
        self.id_to_token = ordered_tokens
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]

    def __len__(self) -> int:
        return len(self.id_to_token)

    @classmethod
    def from_corpora(cls, corpora: Iterable[Iterable[str]]) -> "Vocab":
        tokens: list[str] = []
        for sentences in corpora:
            for sentence in sentences:
                tokens.extend(TOKEN_PATTERN.findall(sentence.lower()))
        return cls(tokens)

    def encode_sentence(self, sentence: str) -> list[int]:
        tokens = TOKEN_PATTERN.findall(sentence.lower())
        return [self.bos_id, *(self.token_to_id[token] for token in tokens), self.eos_id]


class SequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, examples: list[tuple[Tensor, Tensor]], domain_name: str) -> None:
        self.examples = examples
        self.domain_name = domain_name

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.examples[index]

    @classmethod
    def from_sentences(
        cls,
        sentences: list[str],
        vocab: Vocab,
        seq_len: int,
        stride: int,
        domain_name: str,
    ) -> "SequenceDataset":
        stream: list[int] = []
        for sentence in sentences:
            stream.extend(vocab.encode_sentence(sentence))

        window = seq_len + 1
        if len(stream) <= window:
            raise ValueError(f"Not enough tokens to build sequences for {domain_name}.")

        examples: list[tuple[Tensor, Tensor]] = []
        for start in range(0, len(stream) - window + 1, stride):
            segment = stream[start : start + window]
            inputs = torch.tensor(segment[:-1], dtype=torch.long)
            labels = torch.tensor(segment[1:], dtype=torch.long)
            examples.append((inputs, labels))
        return cls(examples, domain_name=domain_name)

    @classmethod
    def from_examples(cls, examples: list[tuple[Tensor, Tensor]], domain_name: str) -> "SequenceDataset":
        return cls(examples=examples, domain_name=domain_name)

    def clone_examples(self) -> list[tuple[Tensor, Tensor]]:
        return [(inputs.clone(), labels.clone()) for inputs, labels in self.examples]


def build_continual_datasets(
    seq_len: int,
    train_sentences_per_domain: int,
    val_sentences_per_domain: int,
    seed: int,
) -> ContinualDatasetBundle:
    old_spec = DomainSpec(
        name="legacy_science",
        subjects=("telescope", "satellite", "rover", "spectrometer", "controller", "archive", "analyst"),
        verbs=("calibrates", "tracks", "compresses", "aligns", "filters", "maps", "reconstructs"),
        objects=("orbit", "nebula", "sensor", "signal", "spectrum", "catalog", "anomaly"),
        descriptors=("stellar", "orbital", "thermal", "distant", "sparse", "lunar", "analog"),
        contexts=("mission", "antenna", "crater", "laboratory", "payload", "horizon", "vacuum"),
        rare_tokens=("quasar", "photon", "apogee", "magnetar"),
    )
    new_spec = DomainSpec(
        name="novel_finance",
        subjects=("broker", "ledger", "auditor", "trader", "regulator", "controller", "archive", "analyst"),
        verbs=("prices", "hedges", "audits", "settles", "flags", "scores", "reallocates"),
        objects=("credit", "option", "balance", "signal", "margin", "invoice", "covenant"),
        descriptors=("liquid", "leveraged", "deferred", "capital", "volatile", "rolling", "sparse"),
        contexts=("market", "hearing", "quarter", "exchange", "filing", "contract", "treasury"),
        rare_tokens=("swaption", "drawdown", "collateral", "basis"),
    )

    old_train = _generate_corpus(old_spec, train_sentences_per_domain, seed + 11)
    old_val = _generate_corpus(old_spec, val_sentences_per_domain, seed + 12)
    new_train = _generate_corpus(new_spec, train_sentences_per_domain, seed + 21)
    new_val = _generate_corpus(new_spec, val_sentences_per_domain, seed + 22)

    vocab = Vocab.from_corpora([old_train, old_val, new_train, new_val])
    stride = max(4, seq_len // 2)

    return ContinualDatasetBundle(
        old_train=SequenceDataset.from_sentences(old_train, vocab, seq_len, stride, old_spec.name),
        old_val=SequenceDataset.from_sentences(old_val, vocab, seq_len, stride, old_spec.name),
        new_train=SequenceDataset.from_sentences(new_train, vocab, seq_len, stride, new_spec.name),
        new_val=SequenceDataset.from_sentences(new_val, vocab, seq_len, stride, new_spec.name),
        vocab=vocab,
        old_name=old_spec.name,
        new_name=new_spec.name,
        sample_sentences={
            old_spec.name: old_train[:5],
            new_spec.name: new_train[:5],
        },
    )


def _generate_corpus(spec: DomainSpec, size: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    shared_terms = ("memory", "pattern", "drift", "cluster", "signal", "vector")
    pacing_terms = ("after", "before", "while", "whenever")
    sentences: list[str] = []

    for _ in range(size):
        subject = rng.choice(spec.subjects)
        verb = rng.choice(spec.verbs)
        obj = rng.choice(spec.objects)
        descriptor = rng.choice(spec.descriptors)
        context = rng.choice(spec.contexts)
        rare = rng.choice(spec.rare_tokens)
        shared = rng.choice(shared_terms)
        pacing = rng.choice(pacing_terms)
        template = rng.randrange(4)

        if template == 0:
            sentence = (
                f"the {descriptor} {subject} {verb} the {obj} near the {context} "
                f"with {shared} {rare}."
            )
        elif template == 1:
            sentence = (
                f"{subject} {verb} each {obj} {pacing} the {context} reveals "
                f"{shared} {descriptor} traces."
            )
        elif template == 2:
            sentence = (
                f"during the {context} the {subject} keeps {shared} {obj} stable "
                f"through {descriptor} {rare} cycles."
            )
        else:
            sentence = (
                f"the {shared} report shows how the {descriptor} {subject} {verb} "
                f"the {obj} across the {context}."
            )

        sentences.append(sentence)
    return sentences
