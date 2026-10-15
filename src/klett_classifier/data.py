"""CSV parsing, normalization, label validation, and leakage checks."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ModelSettings
from .labels import CATEGORIES, LABEL_TO_ID


class DatasetError(ValueError):
    """Invalid input dataset or accidental train/test overlap."""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def load_labeled_csv(path: Path, max_chars: int = ModelSettings().max_query_chars) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"text": "string", "label": "string"})
    if not {"text", "label"}.issubset(frame.columns):
        raise DatasetError("CSV must contain 'text' and 'label' columns")

    optional = [col for col in ("pair_id", "language", "group_id", "source", "family", "kind", "source_record_id", "source_conversation_id", "leakage_group", "split") if col in frame.columns]
    frame = frame[["text", "label", *optional]].dropna(subset=["text", "label"]).copy()
    for col in optional:
        frame[col] = frame[col].astype("string").str.strip()
        if col in {"pair_id", "language", "leakage_group", "split"} and (frame[col].isna().any() or frame[col].eq("").any()):
            raise DatasetError(f"Missing values in {col}")
    if "language" in frame and not set(frame["language"]).issubset({"en", "bg"}):
        raise DatasetError("language must be 'en' or 'bg'")
    frame["text"] = frame["text"].str.strip()
    frame["label"] = frame["label"].str.strip()
    frame = frame[frame["text"].ne("")].copy()
    if frame.empty:
        raise DatasetError("Dataset contains no valid records")

    if frame["text"].str.len().gt(max_chars).any():
        raise DatasetError(f"A query exceeds {max_chars} characters")
    invalid = sorted(set(frame["label"]) - set(CATEGORIES))
    if invalid:
        raise DatasetError(f"Unknown category labels: {invalid}")

    frame["normalized"] = frame["text"].map(normalize_text)
    conflicts = frame.groupby("normalized")["label"].nunique()
    if conflicts.gt(1).any():
        raise DatasetError("Identical normalized texts have contradictory labels")

    # Keep normalized text unique across the full corpus to prevent leakage.
    duplicate_groups = frame.groupby("normalized")
    for normalized, duplicate in duplicate_groups:
        for column in ("pair_id", "group_id"):
            if column in frame.columns and duplicate[column].nunique() > 1:
                raise DatasetError(
                    f"Same normalized text belongs to different {column}s: {normalized[:80]}"
                )
    frame = frame.drop_duplicates(subset=["normalized"]).reset_index(drop=True)
    frame["label_id"] = frame["label"].map(LABEL_TO_ID).astype(int)
    return frame


def split_training_data(frame: pd.DataFrame, seed: int, fraction: float, group_column: str | None = None):
    counts = frame["label"].value_counts()
    if set(counts.index) != set(CATEGORIES) or counts.min() < 5:
        raise DatasetError("Provide at least 5 unique examples for each of the 4 categories")
    if not 0.05 <= fraction <= 0.5:
        raise DatasetError("validation_fraction must be between 0.05 and 0.5")

    if group_column is not None:
        if group_column not in frame.columns:
            raise DatasetError(f"Group column missing: {group_column}")
        # Every group must have exactly one label. Both EN and BG belong to same split.
        group_labels = frame.groupby(group_column)["label_id"].nunique()
        if group_labels.gt(1).any():
            raise DatasetError("Conflicting labels inside group(s)")
        groups = frame.drop_duplicates(group_column)[[group_column, "label_id"]]
        group_counts = groups["label_id"].value_counts()
        if set(group_counts.index) != set(range(len(CATEGORIES))) or group_counts.min() < 5:
            raise DatasetError("Need at least 5 distinct groups per category")
        train_groups, validation_groups = train_test_split(
            groups, test_size=fraction, stratify=groups["label_id"], random_state=seed,
        )
        train = frame[frame[group_column].isin(train_groups[group_column])]
        validation = frame[frame[group_column].isin(validation_groups[group_column])]
        if set(train[group_column]) & set(validation[group_column]):
            raise DatasetError("Groups overlap between training and validation")
    else:
        train, validation = train_test_split(
            frame, test_size=fraction, stratify=frame["label_id"], random_state=seed,
        )
    if set(train["normalized"]) & set(validation["normalized"]):
        raise DatasetError("Training and validation sets overlap")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def save_dataset_manifest(path: Path, frame: pd.DataFrame) -> None:
    """Store fingerprints of all training/validation texts to guard held-out evaluation."""
    payload = {"schema_version": 2, "text_hashes": sorted(fingerprint(x) for x in frame["text"])}
    if "pair_id" in frame.columns:
        payload["pair_ids"] = sorted(set(frame["pair_id"].tolist()))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_independent_test(model_dir: Path, test_frame: pd.DataFrame, test_path: Path | None = None) -> None:
    manifest_path = Path(model_dir) / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise DatasetError("Dataset manifest missing: cannot verify test-set independence")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "held_out_file_sha256" in manifest:
        if test_path is None or hashlib.sha256(Path(test_path).read_bytes()).hexdigest() != manifest["held_out_file_sha256"]:
            raise DatasetError("Held-out test CSV changed or wrong test path; use the locked test.csv")
    if "held_out_text_hashes" in manifest:
        expected = set(manifest["held_out_text_hashes"])
        from .leakage import text_hash
        actual = {text_hash(text) for text in test_frame["text"]}
        if expected != actual:
            raise DatasetError("Test file does not match locked held-out split; use the audited test.csv")
        if "leakage_group" not in test_frame.columns or (
            set(test_frame["leakage_group"]) != set(manifest["held_out_groups"])
        ):
            raise DatasetError("Test leakage_group does not match locked held-out split")
    known = set(manifest["text_hashes"])
    overlapping = sum(fingerprint(text) in known for text in test_frame["text"])
    if overlapping:
        raise DatasetError(f"Test dataset overlaps with training/validation: {overlapping} text(s)")
    if "leakage_group" in test_frame.columns and "leakage_groups" in manifest:
        matches = set(test_frame["leakage_group"]) & set(manifest["leakage_groups"])
        if matches:
            raise DatasetError(f"Test contains {len(matches)} seen leakage_group(s)")
    if "pair_id" in test_frame.columns and "pair_ids" in manifest:
        overlap_pairs = set(test_frame["pair_id"]) & set(manifest["pair_ids"])
        if overlap_pairs:
            raise DatasetError(f"Test contains {len(overlap_pairs)} seen pair_id(s)")
