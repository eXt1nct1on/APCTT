"""Command-line entry point for TRL-Scout."""

from __future__ import annotations

import argparse

from trl_scout.pipeline.run import main as assess_main


def main() -> None:
    """Dispatch supported commands."""
    parser = argparse.ArgumentParser(description="TRL-Scout command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    assess_parser = subparsers.add_parser("assess", help="Run an end-to-end assessment.")
    assess_parser.add_argument("source")
    assess_parser.add_argument("--out", default=None)
    assess_parser.add_argument("--mock", action="store_true")
    assess_parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.command == "assess":
        assess_main_args = [args.source]
        if args.out:
            assess_main_args.extend(["--out", args.out])
        if args.mock:
            assess_main_args.append("--mock")
        if args.no_cache:
            assess_main_args.append("--no-cache")
        import sys

        sys.argv = [sys.argv[0], *assess_main_args]
        assess_main()


if __name__ == "__main__":
    main()
