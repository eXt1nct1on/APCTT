"""Fetch open-access arXiv samples and create evaluation manifest rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

import requests
import yaml

LOGGER = logging.getLogger("trl_scout.fetch_samples")
FIELDS = [
    "doc_id",
    "filename",
    "source_url",
    "doc_type",
    "domain",
    "gold_trl",
    "gold_low",
    "gold_high",
    "labeler_a",
    "labeler_b",
    "notes",
]
USER_AGENT = "TRL-Scout/0.1 (research evaluation; contact: maintainer@example.invalid)"
ATOM = "{http://www.w3.org/2005/Atom}"


def _queries(path: Path) -> list[tuple[str, int]]:
    """Load and validate query/count pairs."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("sample YAML must be a list")
    result: list[tuple[str, int]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("query"):
            raise ValueError("each sample must contain query and count")
        count = int(item.get("count", 1))
        if count < 1:
            raise ValueError("sample count must be positive")
        result.append((str(item["query"]), count))
    return result


def _entries(xml: str) -> list[tuple[str, str]]:
    """Extract arXiv identifiers and PDF links from an Atom response."""
    root = ET.fromstring(xml)
    entries: list[tuple[str, str]] = []
    for entry in root.findall(f"{ATOM}entry"):
        identifier = entry.findtext(f"{ATOM}id", "").rsplit("/", 1)[-1]
        if not identifier:
            continue
        entries.append((identifier, f"https://arxiv.org/pdf/{identifier}.pdf"))
    return entries


def fetch_samples(
    config_path: str | Path = "eval/samples.yaml",
    *,
    docs_dir: str | Path = "eval/gold/docs",
    labels_path: str | Path = "eval/gold/labels.csv",
    session: requests.Session | None = None,
    delay_seconds: float = 3.0,
) -> list[Path]:
    """Download configured arXiv PDFs and append unlabelled manifest rows."""
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    destination = Path(docs_dir)
    destination.mkdir(parents=True, exist_ok=True)
    label_file = Path(labels_path)
    existing: dict[str, dict[str, str]] = {}
    if label_file.exists():
        with label_file.open(newline="", encoding="utf-8") as handle:
            existing = {row["doc_id"]: row for row in csv.DictReader(handle)}
    downloaded: list[Path] = []
    for query, count in _queries(Path(config_path)):
        url = (
            "https://export.arxiv.org/api/query?search_query="
            f"all:{quote_plus(query)}&start=0&max_results={count}"
        )
        response = client.get(url, timeout=15)
        response.raise_for_status()
        for identifier, pdf_url in _entries(response.text)[:count]:
            time.sleep(delay_seconds)
            safe_id = identifier.replace("/", "_")
            output = destination / f"{safe_id}.pdf"
            if not output.exists():
                pdf = client.get(pdf_url, timeout=30)
                pdf.raise_for_status()
                output.write_bytes(pdf.content)
            doc_id = hashlib.sha256(identifier.encode()).hexdigest()[:16]
            existing.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "filename": str(output.relative_to(Path.cwd())),
                    "source_url": pdf_url,
                    "doc_type": "",
                    "domain": "",
                    "gold_trl": "",
                    "gold_low": "",
                    "gold_high": "",
                    "labeler_a": "",
                    "labeler_b": "",
                    "notes": "",
                },
            )
            downloaded.append(output)
    label_file.parent.mkdir(parents=True, exist_ok=True)
    with label_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing.values())
    return downloaded


def main() -> None:
    """Fetch samples from the command line."""
    parser = argparse.ArgumentParser(description="Fetch arXiv evaluation samples.")
    parser.add_argument("--config", default="eval/samples.yaml")
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for path in fetch_samples(args.config, delay_seconds=args.delay):
        LOGGER.info("Fetched %s", path)


if __name__ == "__main__":
    main()
