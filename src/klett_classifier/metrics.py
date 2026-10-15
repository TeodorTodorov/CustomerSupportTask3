"""Consistent classification metrics across training and held-out evaluation."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .labels import CATEGORIES


LABEL_IDS = list(range(len(CATEGORIES)))


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.asarray(logits).argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, labels=LABEL_IDS, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(labels, predictions, labels=LABEL_IDS, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels, predictions, labels=LABEL_IDS, average="macro", zero_division=0)
        ),
    }


def full_report(actual, predicted) -> dict:
    actual = list(actual)
    predicted = list(predicted)
    if not actual:
        raise ValueError("Evaluation dataset must not be empty")
    return {
        "metrics": {
            "accuracy": float(accuracy_score(actual, predicted)),
            "f1_macro": float(f1_score(actual, predicted, labels=LABEL_IDS, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(actual, predicted, labels=LABEL_IDS, average="weighted", zero_division=0)),
        },
        "per_class": classification_report(
            actual,
            predicted,
            labels=LABEL_IDS,
            target_names=CATEGORIES,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABEL_IDS).tolist(),
        "label_order": list(CATEGORIES),
    }


def save_report(report: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
