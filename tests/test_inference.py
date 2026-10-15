import sys
import types

import pytest
import torch

from klett_classifier.config import ModelSettings
from klett_classifier.inference import InvalidQuery, QueryClassifier
from klett_classifier.labels import CATEGORIES


class FakeEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def __call__(self, texts, **kwargs):
        return FakeEncoding({"input_ids": torch.ones((len(texts), 2), dtype=torch.long)})


class FakeModel:
    config = types.SimpleNamespace(id2label=dict(enumerate(CATEGORIES)))

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        logits = torch.tensor([[4.0, 0.0, -1.0, -2.0]] * batch_size)
        return types.SimpleNamespace(logits=logits)


def test_inference_batch_and_validation(tmp_path, monkeypatch):
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        AutoModelForSequenceClassification=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    (tmp_path / "model.safetensors").write_bytes(b"fake-test-safetensors")
    classifier = QueryClassifier(tmp_path, ModelSettings(batch_size=2), device="cpu")
    results = classifier.predict_batch(["  Заявка 1 ", "Заявка 2", "Заявка 3"])
    assert len(results) == 3
    assert all(result.category == "Технически проблем" for result in results)
    assert all(sum(result.scores.values()) == pytest.approx(1.0) for result in results)
    with pytest.raises(InvalidQuery):
        classifier.predict("  ")
    assert classifier.predict_batch([]) == []
