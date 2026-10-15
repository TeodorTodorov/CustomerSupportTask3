"""Generate and translate Klett intent candidates, preserving bilingual provenance.

Run from project root. Uses build_dataset.py (same directory) for EN generation.
Translation is cached in SQLite and resumable across interrupted runs.

IMPORTANT: All outputs are training CANDIDATES, not independently evaluated data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("klett.multilingual")
MODEL = "Helsinki-NLP/opus-mt-en-bg"
LABELS = (
    "Технически проблем", "Достъп или регистрация",
    "Учебно съдържание", "Друго",
)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def text_id(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def read_english(source: Path) -> pd.DataFrame:
    if not source.is_file():
        raise FileNotFoundError(f"English source not found: {source}")
    df = pd.read_csv(source, encoding="utf-8-sig", dtype=str).fillna("")
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("English CSV must contain text,label")
    df = df[df["text"].str.strip().ne("")].copy()
    if not set(df["label"]).issubset(set(LABELS)):
        raise ValueError(f"Unknown labels: {set(df['label']) - set(LABELS)}")
    if df["text"].str.contains(r"[А-Яа-яЍѝ]", regex=True).any():
        raise ValueError("Input file contains Cyrillic; expected source language English")
    df["text"] = df["text"].str.strip()
    df["_normalized"] = df["text"].map(norm)
    conflicts = df.groupby("_normalized")["label"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("English dataset contains contradictory labels for same normalized text")
    df = df.drop_duplicates("_normalized").drop(columns="_normalized").reset_index(drop=True)
    df["pair_id"] = df["text"].map(text_id)
    if not {"source", "family"}.issubset(df.columns):
        provenance_path = source.with_name(source.stem + "_provenance.csv")
        if provenance_path.is_file():
            provenance = pd.read_csv(provenance_path, encoding="utf-8-sig", dtype=str).fillna("")
            if {"text", "source", "family"}.issubset(provenance.columns):
                provenance["_key"] = provenance["text"].map(norm)
                lookup = provenance.drop_duplicates("_key").set_index("_key")
                df["_key"] = df["text"].map(norm)
                for column in ("source", "family", "kind"):
                    if column in lookup.columns:
                        df[column] = df["_key"].map(lookup[column]).fillna("")
                df = df.drop(columns="_key")
    if not {"source", "family"}.issubset(df.columns):
        raise ValueError("Leak-safe pipeline requires source,family columns or *_provenance.csv")
    if df["source"].eq("").any() or df["family"].eq("").any():
        raise ValueError("Missing provenance in some English source rows")
    return df


class TranslationCache:
    def __init__(self, path: Path, model_id: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS translations ("
            "model TEXT NOT NULL, source_id TEXT NOT NULL, "
            "original TEXT NOT NULL, translation TEXT NOT NULL, "
            "PRIMARY KEY(model, source_id))"
        )
        self.connection.commit()
        self.model_id = model_id

    def lookup(self, source: str) -> str | None:
        entry = self.connection.execute(
            "SELECT translation FROM translations WHERE model=? AND source_id=?",
            (self.model_id, text_id(source)),
        ).fetchone()
        return entry[0] if entry else None

    def save_many(self, sources: list[str], translated: list[str]) -> None:
        if len(sources) != len(translated):
            raise ValueError("Translation count does not match input count")
        self.connection.executemany(
            "INSERT OR REPLACE INTO translations(model, source_id, original, translation) "
            "VALUES (?,?,?,?)",
            [(self.model_id, text_id(s), s, t.strip()) for s, t in zip(sources, translated)],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class HelsinkiTranslator:
    def __init__(self, model_id: str, device: str, max_input_tokens: int):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.torch = torch
        self.max_input_tokens = max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device).eval()
        LOG.info("Translation model: %s, device: %s", model_id, self.device)

    def translate(self, texts: list[str], num_beams: int) -> list[str]:
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_input_tokens,
        ).to(self.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**encoded, max_new_tokens=192, num_beams=num_beams)
        return [t.strip() for t in self.tokenizer.batch_decode(ids, skip_special_tokens=True)]


def translate_records(
    english: pd.DataFrame, cache: TranslationCache, translator,
    batch_size: int, num_beams: int,
) -> pd.DataFrame:
    texts = english["text"].tolist()
    missing = [text for text in texts if cache.lookup(text) is None]
    LOG.info("Total: %d, cached: %d, to translate: %d", len(texts), len(texts) - len(missing), len(missing))
    for offset in range(0, len(missing), batch_size):
        batch = missing[offset:offset + batch_size]
        translated = translator.translate(batch, num_beams=num_beams)
        cache.save_many(batch, translated)
        completed = min(offset + len(batch), len(missing))
        if completed % max(batch_size * 10, 1) == 0 or completed == len(missing):
            LOG.info("Translated %d/%d missing items", completed, len(missing))
    result = english.copy()
    result["bg_text"] = [cache.lookup(t) for t in texts]
    return result


def make_outputs(pairs: pd.DataFrame, output_dir: Path, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pairs.copy()
    df["bg_text"] = df["bg_text"].fillna("").astype(str).str.strip()
    df["quality_flag"] = "review_needed"
    invalid = df["bg_text"].eq("") | df["bg_text"].str.len().lt(8)
    # Latin-only or untranslated records need review; no automatic relabelling.
    invalid |= df["bg_text"].str.casefold().eq(df["text"].str.casefold())
    df.loc[invalid, "quality_flag"] = "invalid_translation"
    df["bg_norm"] = df["bg_text"].map(norm)
    usable = df[df["quality_flag"].ne("invalid_translation")].copy()
    conflicting = usable.groupby("bg_norm")["label"].nunique()
    conflict_keys = set(conflicting[conflicting.gt(1)].index)
    df.loc[df["bg_norm"].isin(conflict_keys), "quality_flag"] = "conflicting_translation"
    usable = df[df["quality_flag"].eq("review_needed")].copy()
    # If different English originals become identical Bulgarian translations,
    # preserve one for model CSV and retain every pair in the audit CSV.
    usable = usable.drop_duplicates("bg_norm", keep="first")

    # Includes original, translated, label, identity, and quality signal.
    df[["pair_id", "text", "bg_text", "label", "quality_flag"]].rename(
        columns={"text": "english_text", "bg_text": "bulgarian_text"}
    ).to_csv(output_dir / "bilingual_pairs_review.csv", index=False, encoding="utf-8-sig")
    english_output = df[["text", "label"]].copy().sample(frac=1, random_state=seed)
    english_output.to_csv(output_dir / "english_train_candidates.csv", index=False, encoding="utf-8-sig")
    bulgarian_output = usable[["bg_text", "label"]].rename(columns={"bg_text": "text"})
    bulgarian_output = bulgarian_output.sample(frac=1, random_state=seed)
    bulgarian_output.to_csv(output_dir / "bulgarian_train_candidates.csv", index=False, encoding="utf-8-sig")

    en_multilingual = df[["pair_id", "text", "label", "source", "family"]].copy()
    en_multilingual["language"] = "en"
    bg_multilingual = usable[["pair_id", "bg_text", "label", "source", "family"]].rename(columns={"bg_text": "text"})
    bg_multilingual["language"] = "bg"
    multilingual = pd.concat([en_multilingual, bg_multilingual], ignore_index=True)
    multilingual = multilingual.sample(frac=1, random_state=seed)
    multilingual.to_csv(output_dir / "multilingual_with_pair_ids.csv", index=False, encoding="utf-8-sig")
    # Note: model training code expecting ONLY text,label should NOT randomly split
    # this multilingual file; translations must stay grouped with originals.
    report = {
        "english_rows": int(len(df)),
        "bulgarian_unique_usable_rows": int(len(usable)),
        "invalid_translation_rows": int((df["quality_flag"] == "invalid_translation").sum()),
        "conflicting_translation_rows": int((df["quality_flag"] == "conflicting_translation").sum()),
        "english_labels": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "bulgarian_labels": {str(k): int(v) for k, v in usable["label"].value_counts().items()},
        "caution": (
            "All records are candidates; translation quality and source labels require review. "
            "Do not run random row-wise train/validation splitting on the bilingual dataset. "
            "Keep all items with the same pair_id in the same split; "
            "for templates also group by template family."
        ),
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EN/BG intent classification candidates")
    parser.add_argument("--english-input", type=Path, help="Existing English text,label CSV; skip generation")
    parser.add_argument("--builder", type=Path, default=Path(__file__).with_name("build_dataset.py"))
    parser.add_argument("--target-per-class", type=int, default=3500)
    parser.add_argument("--offline", action="store_true", help="Offline English generation; translation still requires cached model")
    parser.add_argument("--strict", action="store_true", help="Do not silently skip remote dataset failures")
    parser.add_argument("--output-dir", type=Path, default=Path("data/multilingual"))
    parser.add_argument("--cache", type=Path, default=Path("data/translation_cache.sqlite3"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if min(args.target_per_class, args.batch_size, args.beams, args.max_input_tokens) < 1:
        parser.error("Counts, batch-size, beams and max-input-tokens must be positive")

    if args.english_input is None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        english_path = args.output_dir / "source_english.csv"
        cmd = [sys.executable, str(args.builder), "--target-per-class", str(args.target_per_class),
               "--seed", str(args.seed), "--output", str(english_path)]
        if args.offline:
            cmd.append("--offline")
        if args.strict:
            cmd.append("--strict")
        LOG.info("Building English candidates using %s", args.builder)
        subprocess.run(cmd, check=True)
    else:
        english_path = args.english_input

    english = read_english(english_path)
    cache = TranslationCache(args.cache, args.model)
    try:
        missing = sum(cache.lookup(t) is None for t in english["text"])
        if missing:
            translator = HelsinkiTranslator(args.model, args.device, args.max_input_tokens)
        else:
            translator = None
        if translator is not None:
            english = translate_records(english, cache, translator, args.batch_size, args.beams)
        else:
            LOG.info("All translations loaded from cache")
            english["bg_text"] = [cache.lookup(t) for t in english["text"]]
    finally:
        cache.close()
    report = make_outputs(english, args.output_dir, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved files in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
