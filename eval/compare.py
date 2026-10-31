"""Compare evaluation modes in pitch-deck-ready Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(results_dir: str | Path) -> str:
    """Render a Markdown table from mode result directories."""
    root = Path(results_dir)
    rows = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            "| {mode} | {exact:.1%} | {within:.1%} | {mae:.2f} | {spearman:.2f} | "
            "{confidence:.2f} | {calls:.1f} | {seconds:.2f} |".format(
                mode=metrics_path.parent.name,
                exact=metrics["exact_accuracy"],
                within=metrics["within_one_accuracy"],
                mae=metrics["mae"],
                spearman=metrics["spearman"],
                confidence=metrics["mean_confidence_correct"],
                calls=metrics["mean_llm_calls"],
                seconds=metrics["mean_seconds"],
            )
        )
    return (
        "| Mode | Exact | Within ±1 | MAE | Spearman | Confidence (correct) | "
        "LLM calls | Seconds |\n|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
    )


def main() -> None:
    """Print comparison table."""
    parser = argparse.ArgumentParser(description="Compare evaluation modes.")
    parser.add_argument("results_dir", type=Path, default=Path("eval/results"), nargs="?")
    args = parser.parse_args()
    print(compare(args.results_dir))


if __name__ == "__main__":
    main()
