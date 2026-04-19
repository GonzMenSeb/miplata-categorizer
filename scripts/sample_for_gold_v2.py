"""Sample recent user-corrected rows as candidate seeds for gold_set_v2.

Pulls from the categorizer DB (``DATABASE_URL``), joining the ``corrections``
append-only table against ``labeled_transactions`` (which carries the raw
transaction fields) and, best-effort, ``predictions`` (for the confidence
the cascade had when it got the row wrong).

The output is NOT auto-promoted — the operator must review each row
before it becomes ``config/gold_set_v2.jsonl``. The file ships as
``config/gold_set_v2_candidate.jsonl`` for that reason.

Sampling strategy:

  * Filter to corrections from the last ``--days`` days.
  * Bias selection toward slugs NOT present in gold_set_v1.jsonl
    (diversity over redundancy: we already have regression coverage of
    the frequent slugs).
  * Cap at ``--sample`` rows.

Exit codes:
  * 0 — file written (or "no corrections yet" banner + exit 0).
  * 2 — env misconfiguration (missing DATABASE_URL).

Usage (inside the container):
    docker exec categorizer-api python /app/scripts/sample_for_gold_v2.py \\
        --days 30 --sample 30 --out /tmp/gold_set_v2_candidate.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DEFAULT_GOLD_V1 = Path("config/gold_set_v1.jsonl")


def _load_v1_slugs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            slug = obj.get("expected_category_slug")
            if slug:
                out.add(slug)
    return out


async def _fetch_correction_rows(dsn: str, since: datetime) -> list[dict[str, Any]]:
    """Join corrections → labeled_transactions → predictions (LEFT).

    corrections is the ground truth that the user said "you were wrong".
    labeled_transactions carries the raw description/amount/date needed for
    a gold-set row. predictions is joined for the confidence the cascade
    had at prediction time (nullable — old corrections may predate the
    audit).
    """
    eng = create_async_engine(dsn)
    try:
        async with eng.connect() as c:
            rows = await c.execute(
                text(
                    """
                    SELECT
                        c.external_id          AS external_id,
                        c.new_slug             AS new_slug,
                        c.old_slug             AS old_slug,
                        c.created_at           AS corrected_at,
                        lt.account_slug        AS account_slug,
                        lt.tx_date             AS tx_date,
                        lt.amount              AS amount,
                        lt.currency            AS currency,
                        lt.description         AS description,
                        lt.transaction_type    AS transaction_type,
                        (
                            SELECT p.confidence
                            FROM predictions p
                            WHERE p.external_id = c.external_id
                              AND p.predicted_slug = c.old_slug
                            ORDER BY p.created_at DESC
                            LIMIT 1
                        )                      AS confidence_when_corrected
                    FROM corrections c
                    JOIN labeled_transactions lt ON lt.external_id = c.external_id
                    WHERE c.created_at >= :since
                    ORDER BY c.created_at DESC
                    """
                ),
                {"since": since},
            )
            return [dict(r._mapping) for r in rows]
    finally:
        await eng.dispose()


def _diversity_sample(
    rows: list[dict[str, Any]], v1_slugs: set[str], target: int, seed: int = 0xBADA
) -> list[dict[str, Any]]:
    """Pick up to ``target`` rows, biased toward slugs unseen in v1.

    Rows whose ``new_slug`` is NOT already in v1 are sampled first; if we
    still have headroom after taking all of them, fill the remainder with
    a random draw of the rest. Ties broken by recency.
    """
    rng = random.Random(seed)
    novel = [r for r in rows if r["new_slug"] not in v1_slugs]
    familiar = [r for r in rows if r["new_slug"] in v1_slugs]

    rng.shuffle(novel)
    rng.shuffle(familiar)

    picked = novel[:target]
    if len(picked) < target:
        picked += familiar[: target - len(picked)]
    # Deterministic output order: by corrected_at desc.
    picked.sort(key=lambda r: r["corrected_at"], reverse=True)
    return picked


def _format_gold_row(row: dict[str, Any]) -> dict[str, Any]:
    tx_date = row["tx_date"]
    if hasattr(tx_date, "isoformat"):
        tx_date = tx_date.isoformat()
    amount = row["amount"]
    # Numeric comes back as Decimal — json.dumps needs a plain number.
    if hasattr(amount, "__float__"):
        amount = float(amount)
    out: dict[str, Any] = {
        "external_id": row["external_id"],
        "account_slug": row["account_slug"],
        "date": tx_date,
        "amount": amount,
        "currency": row["currency"],
        "description": row["description"],
        "transaction_type": row["transaction_type"],
        "expected_category_slug": row["new_slug"],
    }
    conf = row.get("confidence_when_corrected")
    if conf is not None:
        out["confidence_when_corrected"] = float(conf)
    return out


async def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("config/gold_set_v2_candidate.jsonl"))
    ap.add_argument("--gold-v1", type=Path, default=DEFAULT_GOLD_V1,
                    help="Path to gold_set_v1.jsonl for slug-coverage diffing.")
    args = ap.parse_args(argv)

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2

    since = datetime.now(UTC) - timedelta(days=args.days)
    rows = await _fetch_correction_rows(dsn, since)
    if not rows:
        print(f"no corrections yet — nothing to sample (window: last {args.days} days)")
        return 0

    v1_slugs = _load_v1_slugs(args.gold_v1)
    picked = _diversity_sample(rows, v1_slugs, args.sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(_format_gold_row(r), ensure_ascii=False) + "\n")

    sampled_slugs = Counter(r["new_slug"] for r in picked)
    novel_slugs = {s for s in sampled_slugs if s not in v1_slugs}
    parent_counts = Counter(s.split(".")[0] for s in sampled_slugs)

    print(f"wrote {len(picked)} rows → {args.out}")
    print(f"  sampled from {len(rows)} corrections in the last {args.days} days")
    print(f"  novel slugs vs v1: {len(novel_slugs)}  ({sorted(novel_slugs)})")
    print("  per-parent counts:")
    for parent, n in parent_counts.most_common():
        print(f"    {parent:28s} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
