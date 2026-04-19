"""Thin CLI wrapper around training.dataset.export().

Runs inside the categorizer-api container (scripts/ ships in the image).
Uses only runtime deps (sqlalchemy + pydantic-settings) — no Unsloth.

Usage:
  docker exec categorizer-api python /app/scripts/export_ft_corpus.py \
      --db-url "$DATABASE_URL" --out /tmp/corpus/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make training/ importable when running from /app/scripts/ in the container
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from training.dataset import MIN_CORPUS_WARN, export  # noqa: E402


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Export labeled corpus as ChatML JSONL.")
    p.add_argument("--db-url", default=os.environ.get("DATABASE_URL"), help="Defaults to $DATABASE_URL")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--min-rows", type=int, default=MIN_CORPUS_WARN)
    args = p.parse_args()
    if not args.db_url:
        print("--db-url required (or set DATABASE_URL)", file=sys.stderr)
        return 2
    export(args.db_url, args.out, min_rows=args.min_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
