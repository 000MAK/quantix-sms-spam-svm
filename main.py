#!/usr/bin/env python3
"""Download UCI SMS Spam and train a leakage-safe SVM baseline."""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms%2Bspam%2Bcollection.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_messages(data_dir: Path) -> tuple[list[str], np.ndarray]:
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "SMSSpamCollection"
    if not data_path.exists():
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Quantix-ML-Research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
            data_path.write_bytes(archive.read(member))

    messages: list[str] = []
    labels: list[int] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        label, message = line.split("\t", maxsplit=1)
        labels.append(int(label == "spam"))
        messages.append(message)
    if not messages:
        raise ValueError(f"No SMS records were loaded from {data_path}")
    return messages, np.asarray(labels, dtype=np.int64)


def evaluate(model: Pipeline, messages: list[str], labels: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(messages)
    decision_scores = model.decision_function(messages)
    return {
        "samples": int(labels.size),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_spam": float(precision_score(labels, predictions, zero_division=0)),
        "recall_spam": float(recall_score(labels, predictions, zero_division=0)),
        "f1_spam": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, decision_scores)),
        "confusion_matrix_labels_ham_spam": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def main() -> None:
    args = parse_args()
    messages, labels = load_messages(args.data_dir)
    indexes = np.arange(labels.size)
    train_validation, test = train_test_split(
        indexes, test_size=0.20, random_state=args.seed, stratify=labels
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.20,
        random_state=args.seed,
        stratify=labels[train_validation],
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("svm", LinearSVC(class_weight="balanced", random_state=args.seed)),
        ]
    )
    train_messages = [messages[index] for index in train]
    model.fit(train_messages, labels[train])
    validation_result = evaluate(model, [messages[index] for index in validation], labels[validation])
    test_result = evaluate(model, [messages[index] for index in test], labels[test])

    metrics = {
        "dataset": "UCI SMS Spam Collection",
        "positive_class": "spam",
        "seed": args.seed,
        "split_sizes": {"train": int(train.size), "validation": int(validation.size), "test": int(test.size)},
        "validation": validation_result,
        "test": test_result,
        "source": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "svm_model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Train / validation / test: {train.size} / {validation.size} / {test.size}")
    print(f"Held-out spam F1: {test_result['f1_spam']:.3f}")
    print(f"Held-out ROC-AUC: {test_result['roc_auc']:.3f}")
    print("Metrics saved to outputs/metrics.json")


if __name__ == "__main__":
    main()
