"""Data split and validation tests for mixed EN/BG training."""

import pandas as pd
import pytest

from klett_classifier.data import DatasetError, load_labeled_csv, split_training_data
from klett_classifier.labels import CATEGORIES


def make_pairs(count=12):
    rows = []
    for label_id, category in enumerate(CATEGORIES):
        for pair in range(count):
            for lang in ("en", "bg"):
                rows.append({
                    "text": f"query {label_id} number {pair} language {lang}",
                    "label": category,
                    "pair_id": f"pair-{label_id}-{pair}",
                    "language": lang,
                })
    return pd.DataFrame(rows)


def test_pair_split_stratified_and_disjoint(tmp_path):
    path = tmp_path / "mixed.csv"
    make_pairs().to_csv(path, index=False, encoding="utf-8-sig")
    frame = load_labeled_csv(path)
    train, val = split_training_data(frame, seed=42, fraction=.25, group_column="pair_id")
    assert not set(train.pair_id) & set(val.pair_id)
    assert set(train.language) == set(val.language) == {"en", "bg"}
    assert len(val) == 24
    assert val.groupby("label_id")["pair_id"].nunique().to_dict() == {0: 3, 1: 3, 2: 3, 3: 3}


def test_reject_pair_with_conflicting_labels(tmp_path):
    frame = make_pairs()
    frame.loc[1, "label"] = CATEGORIES[1]
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DatasetError, match="Conflicting labels"):
        split_training_data(load_labeled_csv(path), seed=42, fraction=.2, group_column="pair_id")


def test_reject_duplicate_text_across_pairs(tmp_path):
    frame = make_pairs()
    frame.loc[1, "text"] = frame.loc[0, "text"]
    frame.loc[1, "pair_id"] = "another-pair"
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(DatasetError, match="different pair_id"):
        load_labeled_csv(path)
