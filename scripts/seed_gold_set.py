"""Seed the gold set into the running categorizer via its /v1/label endpoint.

Each row gets POSTed to /v1/label with the expected_category_slug as the
user-confirmed label. This populates the pgvector retrieval index so the
kNN tier has something to retrieve from day 1.

Usage:
    CATEGORIZER_BASE_URL=https://categorizer.web.vespiridion.org \\
    CATEGORIZER_API_TOKEN=... \\
    python scripts/seed_gold_set.py config/gold_set_v1.jsonl

Safe to re-run: the categorizer's external_id uniqueness constraint means
re-posts update rather than duplicate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx


def main(path: Path) -> int:
    base = os.environ["CATEGORIZER_BASE_URL"].rstrip("/")
    token = os.environ["CATEGORIZER_API_TOKEN"]
    url = f"{base}/v1/label"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    with path.open() as f, httpx.Client(timeout=30.0) as client:
        ok = 0
        fail = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            expected = row.pop("expected_category_slug")
            body = {
                "transaction": {
                    "external_id": row["external_id"],
                    "account_slug": row["account_slug"],
                    "date": row["date"],
                    "amount": row["amount"],
                    "currency": row["currency"],
                    "description": row["description"],
                    "transaction_type": row["transaction_type"],
                },
                "category_slug": expected,
            }
            resp = client.post(url, headers=headers, json=body)
            if resp.is_success:
                ok += 1
            else:
                fail += 1
                print(f"FAILED {row['external_id']}: {resp.status_code} {resp.text[:120]}")

    print(f"Seeded {ok} rows ({fail} failed).")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
