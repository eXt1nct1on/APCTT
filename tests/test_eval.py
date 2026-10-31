"""Evaluation harness tests."""

import csv
from pathlib import Path

from eval.metrics import calculate_metrics
from eval.run_eval import mask_trl_mentions, run_evaluation

from trl_scout.config import Settings


def test_mask_removes_explicit_trl_mentions() -> None:
    """Author-stated TRL numbers cannot leak into evaluation input."""
    masked = mask_trl_mentions("The system reached TRL 5; technology readiness level 6.")
    assert "TRL 5" not in masked
    assert "level 6" not in masked.lower()


def test_metrics_calculate_headlines() -> None:
    """Metrics include accuracy, error, confidence, and cost fields."""
    result = calculate_metrics(
        [
            {
                "gold_trl": "4",
                "predicted_trl": "5",
                "confidence": "0.8",
                "quote_verification_rate": "0.5",
                "llm_calls": "3",
                "seconds": "2",
                "doc_type": "paper",
            },
            {
                "gold_trl": "4",
                "predicted_trl": "4",
                "confidence": "0.9",
                "quote_verification_rate": "1",
                "llm_calls": "2",
                "seconds": "1",
                "doc_type": "paper",
            },
        ]
    )
    assert result["exact_accuracy"] == 0.5
    assert result["within_one_accuracy"] == 1.0
    assert result["mae"] == 0.5


def test_fake_documents_run_end_to_end(tmp_path: Path) -> None:
    """The mock harness writes predictions and metrics for all labelled fixtures."""
    labels_path = tmp_path / "labels.csv"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    rows = []
    for index in range(3):
        filename = f"fake-{index}.txt"
        (docs_dir / filename).write_text(
            f"Abstract\nInvented technology {index}.\n\nResults\n"
            "A laboratory test measured performance.\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "doc_id": f"fake-{index}",
                "filename": filename,
                "doc_type": "paper",
                "gold_trl": str(index + 1),
            }
        )
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    output = run_evaluation(
        labels_path,
        output_root=tmp_path / "results",
        settings=Settings(data_dir=str(tmp_path / "data")),
        mock=True,
    )

    assert (output / "predictions.csv").exists()
    assert (output / "metrics.json").exists()
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
