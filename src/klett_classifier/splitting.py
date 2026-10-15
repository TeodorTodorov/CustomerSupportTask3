"""Create immutable train / validation / test splits from provenance components."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import DatasetError, load_labeled_csv
from .labels import CATEGORIES
from .leakage import (GroupingSettings, LeakageError, build_leakage_groups,
                      canonical, text_hash, validate_no_shared_components)


def create_splits(source: Path, destination: Path, *, seed: int = 42,
                  test_fraction: float = .10, validation_fraction: float = .10,
                  settings: GroupingSettings = GroupingSettings(),
                  near_fn=None, semantic_fn=None, approved_review: Path | None = None,
                  checks: dict[str, bool] | None = None) -> dict:
    if test_fraction <= 0 or validation_fraction <= 0 or test_fraction + validation_fraction >= .5:
        raise DatasetError("Require valid positive test/validation fractions with train >= 50%")
    frame = load_labeled_csv(source)
    if "pair_id" not in frame.columns:
        raise DatasetError("pair_id provenance required; prepare bilingual dataset first")
    if "family" not in frame.columns or "source" not in frame.columns:
        raise DatasetError("source and family metadata required for template leakage protection")
    opts = {}
    if near_fn is not None:
        opts["near_fn"] = near_fn
    if semantic_fn is not None:
        opts["semantic_fn"] = semantic_fn
    frame, report = build_leakage_groups(frame, settings, **opts)
    if checks is None:
        checks = {"lexical": near_fn is None, "semantic": semantic_fn is None}
    destination.mkdir(parents=True, exist_ok=True)
    if report["cross_label_near_match_count"]:
        # Deterministic, stable text hashes instead of fragile row indices.
        review_path = destination / "review_conflicts.json"
        review_path.write_text(json.dumps(report["cross_label_near_matches"], ensure_ascii=False, indent=2),
                               encoding="utf-8")
        approved = set()
        if approved_review is not None:
            decisions = json.loads(Path(approved_review).read_text(encoding="utf-8"))
            if not isinstance(decisions, list):
                raise LeakageError("Review decisions must be a JSON array")
            for decision in decisions:
                if decision.get("decision") != "intentional_contrast":
                    raise LeakageError("Only explicit intentional_contrast decisions are accepted")
                pair = decision.get("fingerprints")
                if not isinstance(pair, list) or len(pair) != 2 or any(
                    not isinstance(x, str) or len(x) != 64 for x in pair
                ):
                    raise LeakageError("Review decision has invalid fingerprints")
                approved.add(tuple(sorted(pair)))
        required = {tuple(pair["fingerprints"]) for pair in report["cross_label_near_matches"]}
        if missing := required - approved:
            raise LeakageError(f"{len(missing)} unreviewed cross-label pairs; examine {review_path} "
                               "and explicitly approve intentional contrasts with --approved-review")
    group_rows = frame.groupby("leakage_group", sort=True)
    # If a provenance component contains multiple labels, use label proportions for greedy
    # balancing. A random row-wise split is deliberately forbidden.
    group_records = []
    for key, part in group_rows:
        counts = [int((part["label"] == c).sum()) for c in CATEGORIES]
        group_records.append((key, len(part), counts))
    rng = np.random.default_rng(seed)
    rng.shuffle(group_records)
    group_records.sort(key=lambda x: x[1], reverse=True)
    targets = np.array([1 - test_fraction - validation_fraction, validation_fraction, test_fraction])
    overall_counts = np.array([(frame["label"] == c).sum() for c in CATEGORIES])
    desired = targets[:, None] * overall_counts[None, :]
    count = np.zeros((3, len(CATEGORIES)), dtype=float)
    assign = {}
    split_names = ("train", "validation", "test")
    # Greedy vector balancing, with overshoot penalty.
    for key, size, by_class in group_records:
        vec = np.asarray(by_class, dtype=float)
        scores = []
        for idx in range(3):
            candidate = count.copy()
            candidate[idx] += vec
            class_error = np.square((candidate - desired) / np.maximum(desired, 1)).sum()
            total_error = np.square((candidate.sum(axis=1) - targets * len(frame)) /
                                    np.maximum(targets * len(frame), 1)).sum()
            scores.append(class_error + total_error)
        chosen = int(np.argmin(scores))
        assign[key] = split_names[chosen]
        count[chosen] += vec
    frame["split"] = frame["leakage_group"].map(assign)
    result = {name: frame.loc[frame["split"] == name].copy() for name in split_names}
    for name, part in result.items():
        if part.empty or set(part["label"]) != set(CATEGORIES):
            raise LeakageError(f"{name} split misses categories: insufficient independent groups")
    validate_no_shared_components(result["train"], result["validation"], result["test"])
    hashes = {}
    file_hashes = {}
    for name, part in result.items():
        filename = destination / f"{name}.csv"
        part.to_csv(filename, index=False, encoding="utf-8-sig")
        file_hashes[name] = hashlib.sha256(filename.read_bytes()).hexdigest()
        hashes[name] = sorted(set(part["text"].map(text_hash)))
    manifest = {
        "schema_version": 3, "seed": seed, "source": str(source),
        "similarity_checks_enabled": checks,
        "warning": ("Approximate paraphrase detection is disabled or incomplete; "
                    "do not claim semantic leak-free validation")
                   if not all(checks.values()) else None,
        "parameters": {"test_fraction": test_fraction, "validation_fraction": validation_fraction,
                       **settings.__dict__},
        "split_groups": {name: sorted(set(part["leakage_group"])) for name, part in result.items()},
        "text_hashes": hashes,
        "file_sha256": file_hashes,
        "counts": {name: {"rows": len(part),
                          "labels": {c: int((part["label"] == c).sum()) for c in CATEGORIES},
                          "languages": part["language"].value_counts().to_dict() if "language" in part else {}}
                   for name, part in result.items()},
        "limitations": "Approximate near/semantic detection is not a mathematical guarantee; hold out independent natural queries.",
        "audit": {k: v for k, v in report.items() if k != "cross_label_near_matches"},
    }
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_split_files(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    directory = Path(directory)
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for name in ("train", "validation", "test"):
        filename = directory / f"{name}.csv"
        digest = hashlib.sha256(filename.read_bytes()).hexdigest()
        if manifest.get("file_sha256", {}).get(name) != digest:
            raise LeakageError(f"{name}.csv changed since leakage audit (SHA256 mismatch)")
        frame = load_labeled_csv(filename)
        if "leakage_group" not in frame:
            raise LeakageError("Split CSV missing leakage_group")
        if set(frame["text"].map(text_hash)) != set(manifest["text_hashes"][name]):
            raise LeakageError(f"{name}.csv changed since leakage audit; rerun splitting")
        if set(frame["leakage_group"]) != set(manifest["split_groups"][name]):
            raise LeakageError(f"{name}.csv group assignments changed")
        frames[name] = frame
    validate_no_shared_components(*(frames[n] for n in ("train", "validation", "test")))
    return frames["train"], frames["validation"], frames["test"], manifest
