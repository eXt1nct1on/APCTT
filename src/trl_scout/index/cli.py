"""Command-line entry point for hybrid document retrieval."""

import argparse

from trl_scout.index.retriever import DocIndex
from trl_scout.ingest.loaders import load_document


def main() -> None:
    """Load a document and print its top matching chunks."""
    parser = argparse.ArgumentParser(description="Search an ingested document.")
    parser.add_argument("source", help="Document path, URL, arXiv id, or raw text.")
    parser.add_argument("query", help="Search query.")
    parser.add_argument("-k", type=int, default=8, help="Number of chunks to print.")
    args = parser.parse_args()
    document = load_document(args.source)
    results = DocIndex(document).search(args.query, k=args.k)
    for rank, (chunk, score) in enumerate(results, 1):
        print(f"{rank}. [{chunk.chunk_id} p.{chunk.page_start} {chunk.section}] {score:.5f}")
        print(chunk.text)


if __name__ == "__main__":
    main()
