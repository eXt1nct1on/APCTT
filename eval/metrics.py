"""Metrics for TRL-Scout evaluation predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _spearman(actual: list[float], predicted: list[float]) -> float:
    """Calculate Spearman correlation without requiring scipy."""
    if len(actual) < 2:
        return 0.0

    def ranks(values: list[float]) -> list[float]:
        ordered = sorted(enumerate(values), key=lambda pair: pair[1])
        result = [0.0] * len(values)
        index = 0
        while index < len(ordered):
            end = index
            while end < len(ordered) and ordered[end][1] == ordered[index][1]:
                end += 1
            rank = (index + end - 1) / 2 + 1
            for position in range(index, end):
                result[ordered[position][0]] = rank
            index = end
        return result

    left, right = ranks(actual), ranks(predicted)
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else 0.0


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate headline, quality, cost, and breakdown metrics."""
    valid = [row for row in rows if row.get("gold_trl") not in ("", None)]
    actual = [float(row["gold_trl"]) for row in valid]
    predicted = [float(row["predicted_trl"]) for row in valid]
    exact = sum(a == p for a, p in zip(actual, predicted, strict=True))
    within = sum(abs(a - p) <= 1 for a, p in zip(actual, predicted, strict=True))
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_type[str(row.get("doc_type", "unknown"))].append(row)
    breakdown = {}
    for doc_type, group in by_type.items():
        breakdown[doc_type] = {
            "count": len(group),
            "exact_accuracy": mean(
                float(item["gold_trl"]) == float(item["predicted_trl"]) for item in group
            ),
            "mae": mean(
                abs(float(item["gold_trl"]) - float(item["predicted_trl"])) for item in group
            ),
        }
    correct_confidence = [
        float(row["confidence"])
        for row in valid
        if float(row["gold_trl"]) == float(row["predicted_trl"])
    ]
    wrong_confidence = [
        float(row["confidence"])
        for row in valid
        if float(row["gold_trl"]) != float(row["predicted_trl"])
    ]
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for actual_value, predicted_value in zip(actual, predicted, strict=True):
        matrix[str(int(actual_value))][str(int(predicted_value))] += 1
    return {
        "count": len(valid),
        "exact_accuracy": exact / len(valid) if valid else 0.0,
        "within_one_accuracy": within / len(valid) if valid else 0.0,
        "mae": (
            mean(abs(a - p) for a, p in zip(actual, predicted, strict=True))
            if valid
            else 0.0
        ),
        "spearman": _spearman(actual, predicted),
        "confusion_matrix": {key: dict(value) for key, value in matrix.items()},
        "doc_type_breakdown": breakdown,
        "mean_confidence_correct": mean(correct_confidence) if correct_confidence else 0.0,
        "mean_confidence_wrong": mean(wrong_confidence) if wrong_confidence else 0.0,
        "quote_verification_rate_before_filtering": mean(
            float(row.get("quote_verification_rate", 0)) for row in valid
        )
        if valid
        else 0.0,
        "mean_llm_calls": mean(float(row.get("llm_calls", 0)) for row in valid) if valid else 0.0,
        "mean_seconds": mean(float(row.get("seconds", 0)) for row in valid) if valid else 0.0,
    }


def write_metrics(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    """Write metrics JSON and a confusion matrix PNG."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = calculate_metrics(rows)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        labels = sorted(
            {int(row["gold_trl"]) for row in rows if row.get("gold_trl") not in ("", None)}
        )
        labels = sorted(set(labels) | {int(row["predicted_trl"]) for row in rows})
        actual = [int(row["gold_trl"]) for row in rows if row.get("gold_trl") not in ("", None)]
        predicted = [
            int(row["predicted_trl"]) for row in rows if row.get("gold_trl") not in ("", None)
        ]
        display = ConfusionMatrixDisplay(
            confusion_matrix(actual, predicted, labels=labels), display_labels=labels
        )
        display.plot(cmap="Blues", values_format="d")
        plt.tight_layout()
        plt.savefig(destination / "confusion_matrix.png", dpi=160)
        plt.close()
    except ImportError:
        pass
    return metrics


def main() -> None:
    """Calculate metrics from a predictions CSV."""
    import csv

    parser = argparse.ArgumentParser(description="Calculate evaluation metrics.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    write_metrics(rows, args.out or args.predictions.parent)


if __name__ == "__main__":
    main()
