import json

import pandas as pd
import pytest

from klett_classifier.data import DatasetError, assert_independent_test
from klett_classifier.labels import CATEGORIES
from klett_classifier.leakage import GroupingSettings, LeakageError, text_hash
from klett_classifier.splitting import create_splits, validate_split_files


def no_semantic(*args):
    return []


def frame_data():
    items = []
    for idx, label in enumerate(CATEGORIES):
        for n in range(30):
            for lang in ("en", "bg"):
                items.append({
                    "text": f"{lang} example class {idx} number {n} details",
                    "label": label,
                    "language": lang,
                    "pair_id": f"{idx}-{n}",
                    "source": "independent",
                    "family": f"{idx}-{n}",
                })
    return pd.DataFrame(items)


def test_manifest_blocks_modified_label_and_test_file(tmp_path):
    source = tmp_path / "original.csv"
    frame_data().to_csv(source, index=False, encoding="utf-8-sig")
    dest = tmp_path / "splits"
    manifest = create_splits(source, dest, near_fn=lambda *args: [],
                             semantic_fn=no_semantic,
                             settings=GroupingSettings(max_component_fraction=.3))
    assert manifest["file_sha256"]["test"]
    train, val, test, _ = validate_split_files(dest)
    assert {"en", "bg"}.issubset(set(test.language))
    assert not (set(train.pair_id) & set(val.pair_id))
    assert not (set(train.pair_id) & set(test.pair_id))
    test.loc[test.index[0], "label"] = CATEGORIES[3]  # tamper only label
    test.to_csv(dest / "test.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(LeakageError, match="SHA256 mismatch"):
        validate_split_files(dest)


def test_human_review_keeps_contrasts_same_split(tmp_path):
    frame = frame_data()
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False, encoding="utf-8-sig")
    dest = tmp_path / "splits"

    def near(texts, threshold):
        return [(0, 60, .95)]  # Different labels; review is mandatory.

    with pytest.raises(LeakageError, match="unreviewed"):
        create_splits(source, dest, near_fn=near, semantic_fn=no_semantic,
                      settings=GroupingSettings(max_component_fraction=.3))
    pending = json.loads((dest / "review_conflicts.json").read_text(encoding="utf-8"))
    assert len(pending) == 1
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps([{"fingerprints": pending[0]["fingerprints"],
                                    "decision": "intentional_contrast"}]), encoding="utf-8")
    create_splits(source, dest, near_fn=near, semantic_fn=no_semantic,
                  approved_review=approved,
                  settings=GroupingSettings(max_component_fraction=.3))
    train, val, test, _ = validate_split_files(dest)
    candidate_groups = {}
    for split_name, part in (("train", train), ("validation", val), ("test", test)):
        for _, row in part.iterrows():
            if text_hash(row.text) in {text_hash(frame.iloc[0].text), text_hash(frame.iloc[60].text)}:
                candidate_groups.setdefault(split_name, 0)
                candidate_groups[split_name] += 1
    assert list(candidate_groups.values()) == [2]


def test_pair_label_conflict_rejected(tmp_path):
    frame = frame_data()
    frame.loc[1, "label"] = CATEGORIES[3]
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(LeakageError, match="contradictory"):
        create_splits(path, tmp_path / "out", near_fn=lambda *a: [],
                      semantic_fn=no_semantic, settings=GroupingSettings(max_component_fraction=.3))
