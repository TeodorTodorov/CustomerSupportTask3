"""Strict evaluation on an independent test CSV."""

from pathlib import Path

from .data import assert_independent_test, load_labeled_csv
from .inference import QueryClassifier
from .labels import LABEL_TO_ID
from .metrics import full_report, save_report


def evaluate(model_dir: Path, csv_path: Path, report_path: Path | None = None) -> dict:
    frame = load_labeled_csv(csv_path)
    assert_independent_test(model_dir, frame, csv_path)
    classifier = QueryClassifier(model_dir)
    predicted = []
    texts = frame["text"].tolist()
    for start in range(0, len(texts), 64):
        predicted.extend(
            LABEL_TO_ID[result.category]
            for result in classifier.predict_batch(texts[start : start + 64])
        )
    report = full_report(frame["label_id"].tolist(), predicted)
    if "language" in frame.columns:
        report["by_language"] = {}
        for language, part in frame.groupby("language"):
            indices = part.index.tolist()
            report["by_language"][str(language)] = {
                "count": len(indices),
                **full_report(
                    part["label_id"].tolist(),
                    [predicted[i] for i in indices],
                ),
            }
    if report_path:
        save_report(report, report_path)
    return report
