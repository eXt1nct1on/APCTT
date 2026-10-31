"""Command-line entry point for document ingestion."""

import argparse

from trl_scout.ingest.loaders import load_document


def main() -> None:
    """Load a source and print a compact ingestion summary."""
    parser = argparse.ArgumentParser(description="Ingest a TRL-Scout source document.")
    parser.add_argument("source", help="Path, URL, arXiv id, or raw text.")
    args = parser.parse_args()
    document = load_document(args.source)
    sections = sorted({chunk.section for chunk in document.chunks})
    print(f"sections: {', '.join(sections)}")
    print(f"chunks: {len(document.chunks)}")
    if document.chunks:
        print(f"first chunk: {document.chunks[0].text}")


if __name__ == "__main__":
    main()
