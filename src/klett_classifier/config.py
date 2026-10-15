"""Immutable application settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSettings:
    base_model: str = "FacebookAI/xlm-roberta-base"
    max_tokens: int = 192
    max_query_chars: int = 4000
    batch_size: int = 16


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    validation_fraction: float = 0.2
    seed: int = 42
    output_dir: Path = Path("models/classifier")
