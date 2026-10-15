"""Reusable, batched inference layer independent of training and HTTP."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .config import ModelSettings
from .labels import CATEGORIES


class InvalidQuery(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Prediction:
    category: str
    confidence: float
    scores: dict[str, float]
    method: str = "xlm-roberta"

    def to_dict(self) -> dict:
        return asdict(self)


class QueryClassifier:
    def __init__(self, model_dir: Path, settings: ModelSettings | None = None, device: str = "auto"):
        self.settings = settings or ModelSettings()
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Trained model directory not found: {model_dir}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        # Binary torch checkpoints use pickle. Only safetensors artifacts are
        # accepted at inference, and no remote code or downloads are allowed.
        if not ((model_dir / "model.safetensors").is_file() or
                (model_dir / "model.safetensors.index.json").is_file()):
            raise ValueError("Only local safetensors model artifacts are allowed")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True, trust_remote_code=False
        ).to(self.device)
        self.model.eval()

        labels = tuple(self.model.config.id2label[i] for i in range(len(CATEGORIES)))
        if labels != CATEGORIES:
            raise ValueError(f"Model category mapping mismatch: {labels}")

    def _validate(self, text: str) -> str:
        from .security import validate_query
        return validate_query(text, self.settings.max_query_chars)

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        if not isinstance(texts, list):
            raise InvalidQuery("Expected a list of queries")
        if not texts:
            return []
        clean = [self._validate(text) for text in texts]
        predictions = []
        for start in range(0, len(clean), self.settings.batch_size):
            batch = clean[start : start + self.settings.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.settings.max_tokens,
                return_tensors="pt",
            ).to(self.device)
            scores = self.model(**inputs).logits.softmax(dim=-1).cpu().tolist()
            for probabilities in scores:
                best = max(range(len(probabilities)), key=probabilities.__getitem__)
                predictions.append(
                    Prediction(
                        category=CATEGORIES[best],
                        confidence=float(probabilities[best]),
                        scores={name: float(p) for name, p in zip(CATEGORIES, probabilities)},
                    )
                )
        return predictions

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]
