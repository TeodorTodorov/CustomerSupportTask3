"""Reproducible fine-tuning with best checkpoint chosen by validation Macro-F1."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import ModelSettings, TrainingSettings
from .splitting import validate_split_files
from .labels import CATEGORIES, ID_TO_LABEL, LABEL_TO_ID
from .metrics import compute_metrics, full_report, save_report

LOGGER = logging.getLogger(__name__)


class TokenizedDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_tokens: int):
        self.tokens = tokenizer(texts, max_length=max_tokens, truncation=True)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {**{key: val[index] for key, val in self.tokens.items()}, "labels": self.labels[index]}


def train(split_dir: Path, training: TrainingSettings, model: ModelSettings | None = None) -> dict:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    model = model or ModelSettings()
    if training.epochs <= 0 or training.batch_size <= 0 or training.learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")

    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training.seed)

    train_frame, validation_frame, test_frame, split_manifest = validate_split_files(split_dir)
    frame = None  # Never select hyperparameters using held-out test data.
    group_column = "leakage_group"
    output = Path(training.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Dataset: %d total, %d train, %d validation", len(train_frame) + len(validation_frame) + len(test_frame), len(train_frame), len(validation_frame))

    tokenizer = AutoTokenizer.from_pretrained(model.base_model)
    model_instance = AutoModelForSequenceClassification.from_pretrained(
        model.base_model,
        num_labels=len(CATEGORIES),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    train_ds = TokenizedDataset(
        train_frame["text"].tolist(), train_frame["label_id"].tolist(), tokenizer, model.max_tokens
    )
    validation_ds = TokenizedDataset(
        validation_frame["text"].tolist(),
        validation_frame["label_id"].tolist(),
        tokenizer,
        model.max_tokens,
    )

    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=training.epochs,
        per_device_train_batch_size=training.batch_size,
        per_device_eval_batch_size=training.batch_size,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        save_safetensors=True,
        logging_steps=10,
        logging_strategy="steps",
        report_to="none",
        seed=training.seed,
        data_seed=training.seed,
    )
    trainer = Trainer(
        model=model_instance,
        args=args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    prediction = trainer.predict(validation_ds)
    predicted = np.asarray(prediction.predictions).argmax(axis=-1)
    report = full_report(validation_frame["label_id"].tolist(), predicted.tolist())
    report["counts"] = {"train": len(train_frame), "validation": len(validation_frame)}
    report["group_column"] = group_column
    if group_column:
        report["unique_groups"] = {
            "train": int(train_frame[group_column].nunique()),
            "validation": int(validation_frame[group_column].nunique()),
        }
    if "language" in validation_frame.columns:
        languages = validation_frame["language"].to_numpy()
        y_true = validation_frame["label_id"].to_numpy()
        report["by_language"] = {}
        for language in sorted(set(languages)):
            mask = languages == language
            report["by_language"][language] = {
                "count": int(mask.sum()),
                **full_report(y_true[mask].tolist(), predicted[mask].tolist()),
            }
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    save_report(report, output / "validation_report.json")
    import json
    (output / "dataset_manifest.json").write_text(
        json.dumps({"schema_version": 3,
                    "text_hashes": split_manifest["text_hashes"]["train"] + split_manifest["text_hashes"]["validation"],
                    "pair_ids": sorted(set(train_frame.get("pair_id", [])) | set(validation_frame.get("pair_id", []))),
                    "leakage_groups": sorted(set(train_frame["leakage_group"]) | set(validation_frame["leakage_group"])),
                    "held_out_text_hashes": split_manifest["text_hashes"]["test"],
                    "held_out_groups": split_manifest["split_groups"]["test"],
                    "held_out_file_sha256": split_manifest["file_sha256"]["test"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Best checkpoint saved at %s; validation Macro-F1: %.4f", output, report["metrics"]["f1_macro"])
    return report
