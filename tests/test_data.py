import pandas as pd
import pytest

from klett_classifier.data import (
    DatasetError,
    assert_independent_test,
    load_labeled_csv,
    normalize_text,
    save_dataset_manifest,
    split_training_data,
)
from klett_classifier.labels import CATEGORIES


def test_normalization_handles_case_and_spaces():
    assert normalize_text("  ЗАБРАВИХ   парола\n") == "забравих парола"


def test_csv_deduplicates_normalized_text(tmp_path):
    path = tmp_path / "queries.csv"
    pd.DataFrame({"text": ["Грешка  в сайта", "грешка в сайта"], "label": [CATEGORIES[0]] * 2}).to_csv(path, index=False)
    assert len(load_labeled_csv(path)) == 1


def test_rejects_conflicting_labels(tmp_path):
    path = tmp_path / "queries.csv"
    pd.DataFrame({"text": ["Проблем", " проблем "], "label": CATEGORIES[:2]}).to_csv(path, index=False)
    with pytest.raises(DatasetError, match="contradictory"):
        load_labeled_csv(path)


def test_train_validation_disjoint():
    frame = pd.DataFrame(
        {"text": [f"example {i} class {k}" for k in range(4) for i in range(10)],
         "normalized": [f"example {i} class {k}" for k in range(4) for i in range(10)],
         "label": [CATEGORIES[k] for k in range(4) for _ in range(10)],
         "label_id": [k for k in range(4) for _ in range(10)]}
    )
    train, val = split_training_data(frame, 42, 0.2)
    assert not (set(train["normalized"]) & set(val["normalized"]))
    assert len(train) == 32 and len(val) == 8


def test_test_set_overlap_is_rejected(tmp_path):
    train = pd.DataFrame({"text": ["Забравих паролата"]})
    save_dataset_manifest(tmp_path / "dataset_manifest.json", train)
    test = pd.DataFrame({"text": ["  ЗАБРАВИХ   ПАРОЛАТА "]})
    with pytest.raises(DatasetError, match="overlaps"):
        assert_independent_test(tmp_path, test)
