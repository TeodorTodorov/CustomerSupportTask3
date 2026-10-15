"""Exercise the real training orchestration with fake Hugging Face components.

This does not test pretrained weights, but it detects CLI/split/trainer API wiring bugs.
"""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from klett_classifier.config import TrainingSettings
from klett_classifier.labels import CATEGORIES, LABEL_TO_ID
from klett_classifier.splitting import create_splits
from klett_classifier.leakage import GroupingSettings


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[1, 2]] * len(texts), "attention_mask": [[1, 1]] * len(texts)}

    def save_pretrained(self, path):
        return None


class FakeModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeTrainingArguments:
    def __init__(self, **kwargs):
        assert kwargs["eval_strategy"] == "epoch"
        assert kwargs["save_safetensors"] is True
        assert kwargs["metric_for_best_model"] == "f1_macro"
        self.kwargs = kwargs


class FakeTrainer:
    def __init__(self, **kwargs):
        self.validation = kwargs["eval_dataset"]
        self.compute_metrics = kwargs["compute_metrics"]

    def train(self):
        return None

    def predict(self, dataset):
        labels = np.array(dataset.labels)
        logits = np.full((len(labels), 4), -5.0)
        logits[np.arange(len(labels)), labels] = 5.0
        assert self.compute_metrics((logits, labels))["f1_macro"] == 1.0
        return types.SimpleNamespace(predictions=logits)

    def save_model(self, directory):
        from pathlib import Path
        Path(directory, "model.safetensors").write_bytes(b"mock weights, not actual checkpoint")


class FakeEarlyStop:
    def __init__(self, early_stopping_patience):
        assert early_stopping_patience == 2


class FakeCollator:
    def __init__(self, tokenizer):
        pass


def fixture_dataset(tmp_path):
    rows = []
    for label_index, label in enumerate(CATEGORIES):
        for i in range(25):
            for lang in ("en", "bg"):
                rows.append({"text": f"{lang} message kind {label_index} unique number {i}",
                             "label": label, "language": lang, "source": "unit",
                             "family": f"{label_index}-{i}", "pair_id": f"{label_index}-{i}"})
    source = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8-sig")
    split = tmp_path / "splits"
    create_splits(source, split, settings=GroupingSettings(max_component_fraction=0.5),
                  near_fn=lambda *args: [], semantic_fn=lambda *args: [])
    return split


def test_training_pipeline_and_test_manifest(tmp_path, monkeypatch):
    split = fixture_dataset(tmp_path)
    fake_module = types.ModuleType("transformers")
    fake_module.AutoModelForSequenceClassification = FakeModel
    fake_module.AutoTokenizer = FakeTokenizer
    fake_module.DataCollatorWithPadding = FakeCollator
    fake_module.EarlyStoppingCallback = FakeEarlyStop
    fake_module.Trainer = FakeTrainer
    fake_module.TrainingArguments = FakeTrainingArguments
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    from klett_classifier.training import train
    output = tmp_path / "trained"
    report = train(split, TrainingSettings(output_dir=output, epochs=1))
    assert report["metrics"]["f1_macro"] == 1.0
    assert report["by_language"]["bg"]["metrics"]["f1_macro"] == 1.0
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["held_out_file_sha256"]) == 64
    assert (output / "model.safetensors").is_file()

    # Validation and train rows are never used as the held-out test.
    from klett_classifier.data import assert_independent_test, load_labeled_csv, DatasetError
    actual_test = load_labeled_csv(split / "test.csv")
    assert_independent_test(output, actual_test, split / "test.csv")
    with pytest.raises(DatasetError, match="Held-out test CSV changed"):
        assert_independent_test(output, actual_test, split / "train.csv")


def test_metric_consistency():
    from klett_classifier.metrics import full_report, compute_metrics
    actual = [0, 0, 1, 1, 2, 2, 3, 3]
    pred = [0, 0, 1, 1, 2, 2, 3, 3]
    report = full_report(actual, pred)
    assert report["metrics"]["f1_macro"] == 1.0
    assert report["confusion_matrix"] == [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]
    logits = np.eye(4)[actual]
    assert compute_metrics((logits, np.array(actual)))["f1_macro"] == 1.0


def test_cli_predict_rules(capsys):
    from klett_classifier.cli import main
    assert main(["predict", "--backend", "rules", "--text", "Видеоурокът не се зарежда."]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["category"] == "Технически проблем"
    assert result["method"] == "rules"


def test_evaluation_model_bound_test_hash(tmp_path, monkeypatch):
    split = fixture_dataset(tmp_path)
    # Prepare minimal manifests and a classifier that predicts labels from the text.
    from klett_classifier.splitting import validate_split_files
    train, validation, test, manifest = validate_split_files(split)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "dataset_manifest.json").write_text(json.dumps({
        "text_hashes": manifest["text_hashes"]["train"] + manifest["text_hashes"]["validation"],
        "leakage_groups": manifest["split_groups"]["train"] + manifest["split_groups"]["validation"],
        "held_out_text_hashes": manifest["text_hashes"]["test"],
        "held_out_groups": manifest["split_groups"]["test"],
        "held_out_file_sha256": manifest["file_sha256"]["test"],
    }), encoding="utf-8")

    class FakeClassifierForEvaluation:
        def __init__(self, path):
            pass

        def predict_batch(self, texts):
            return [types.SimpleNamespace(category=CATEGORIES[int(text.split("kind ")[1][0])])
                    for text in texts]

    import klett_classifier.evaluation as evaluation
    monkeypatch.setattr(evaluation, "QueryClassifier", FakeClassifierForEvaluation)
    result = evaluation.evaluate(model_dir, split / "test.csv", model_dir / "test_report.json")
    assert result["metrics"]["f1_macro"] == 1.0
    assert result["by_language"]["en"]["metrics"]["f1_macro"] == 1.0
