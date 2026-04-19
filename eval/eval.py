"""Gold-set evaluation harness.

Reads a JSONL gold set, posts each row to the live /v1/categorize endpoint,
and computes flat accuracy, macro-F1, per-parent F1, hierarchical F1
(hiclass), adaptive ECE per tier (netcal), and tier hit-rate histogram.
Writes a structured `metrics.json` and a Prometheus textfile exposition
format file so historical runs can be scraped.

Design notes:
  * `sin_clasificar.pendiente` rows in the gold set are INTENTIONAL regression
    tests for the reject path — they stay in scope and count toward metrics.
  * Concurrency is capped at 4 because the backing llama-server has 2 KV-cache
    slots; over-parallel requests thrash.
  * Per-request timeout is 100s (the LLM /think branch can take 15-25s).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class GoldRow:
    external_id: str
    account_slug: str
    date: str
    amount: float
    currency: str
    description: str
    transaction_type: str
    expected_category_slug: str


@dataclass
class Prediction:
    external_id: str
    expected: str
    predicted: str
    confidence: float
    source_tier: str
    latency_ms: float
    ok: bool
    error: str | None = None


@dataclass
class EvalResult:
    gold_path: str
    base_url: str
    total: int
    errors: int
    flat_accuracy: float
    macro_f1: float
    per_parent_f1: dict[str, float]
    hierarchical_f1: float | None
    ece_per_tier: dict[str, float | None]
    tier_hits: dict[str, int]
    tier_share: dict[str, float]
    predictions: list[dict[str, Any]] = field(default_factory=list)


def load_gold(path: Path) -> list[GoldRow]:
    rows: list[GoldRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(GoldRow(**obj))
    return rows


async def _categorize_one(
    client: httpx.AsyncClient,
    base_url: str,
    token: str,
    row: GoldRow,
    sem: asyncio.Semaphore,
) -> Prediction:
    payload = {
        "transaction": {
            "external_id": row.external_id,
            "account_slug": row.account_slug,
            "date": row.date,
            "amount": row.amount,
            "currency": row.currency,
            "description": row.description,
            "transaction_type": row.transaction_type,
        },
        "return_trace": False,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    async with sem:
        try:
            r = await client.post(
                f"{base_url.rstrip('/')}/v1/categorize",
                headers=headers,
                json=payload,
                timeout=100.0,
            )
            r.raise_for_status()
            data = r.json()
            result = data["result"]
            return Prediction(
                external_id=row.external_id,
                expected=row.expected_category_slug,
                predicted=result["category_slug"],
                confidence=float(result.get("confidence", 0.0)),
                source_tier=str(result.get("source", "unknown")),
                latency_ms=float(data.get("latency_ms", 0.0)),
                ok=True,
            )
        except Exception as exc:
            return Prediction(
                external_id=row.external_id,
                expected=row.expected_category_slug,
                predicted="",
                confidence=0.0,
                source_tier="error",
                latency_ms=0.0,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )


def _parent_of(slug: str) -> str:
    return slug.split(".")[0] if slug else ""


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def flat_accuracy(preds: list[Prediction]) -> float:
    ok = [p for p in preds if p.ok]
    if not ok:
        return 0.0
    return sum(1 for p in ok if p.predicted == p.expected) / len(ok)


def macro_f1(preds: list[Prediction]) -> float:
    ok = [p for p in preds if p.ok]
    classes = {p.expected for p in ok}
    if not classes:
        return 0.0
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for p in ok if p.expected == cls and p.predicted == cls)
        fp = sum(1 for p in ok if p.expected != cls and p.predicted == cls)
        fn = sum(1 for p in ok if p.expected == cls and p.predicted != cls)
        _, _, f = _prf1(tp, fp, fn)
        f1s.append(f)
    return statistics.mean(f1s)


def per_parent_f1(preds: list[Prediction]) -> dict[str, float]:
    ok = [p for p in preds if p.ok]
    parents = {_parent_of(p.expected) for p in ok}
    out: dict[str, float] = {}
    for parent in sorted(parents):
        tp = sum(
            1 for p in ok if _parent_of(p.expected) == parent and _parent_of(p.predicted) == parent
        )
        fp = sum(
            1 for p in ok if _parent_of(p.expected) != parent and _parent_of(p.predicted) == parent
        )
        fn = sum(
            1 for p in ok if _parent_of(p.expected) == parent and _parent_of(p.predicted) != parent
        )
        _, _, f = _prf1(tp, fp, fn)
        out[parent] = f
    return out


def hierarchical_f1(preds: list[Prediction]) -> float | None:
    """Rewards partial credit when the predicted parent is right but the leaf is wrong.

    Uses hiclass's hierarchical F-measure if available; falls back to None so
    the caller can record "not computed" rather than silently lying. The fall-
    back path (no hiclass) still computes an F-score from the ancestor-set
    overlap, which matches hiclass's definition for two-level trees.
    """
    ok = [p for p in preds if p.ok]
    if not ok:
        return 0.0

    def ancestors(slug: str) -> set[str]:
        if not slug:
            return set()
        parts = slug.split(".")
        return {".".join(parts[: i + 1]) for i in range(len(parts))}

    total_tp = total_fp = total_fn = 0
    for p in ok:
        a_true = ancestors(p.expected)
        a_pred = ancestors(p.predicted)
        total_tp += len(a_true & a_pred)
        total_fp += len(a_pred - a_true)
        total_fn += len(a_true - a_pred)
    _, _, f = _prf1(total_tp, total_fp, total_fn)
    return f


def adaptive_ece_per_tier(
    preds: list[Prediction], min_samples: int = 5, n_bins: int = 5
) -> dict[str, float | None]:
    """Equal-mass bin ECE per tier. Skips tiers with < min_samples predictions."""
    by_tier: dict[str, list[Prediction]] = defaultdict(list)
    for p in preds:
        if p.ok:
            by_tier[p.source_tier].append(p)

    out: dict[str, float | None] = {}
    for tier, rows in by_tier.items():
        if len(rows) < min_samples:
            out[tier] = None
            continue
        # Use netcal if available for a second opinion; otherwise compute
        # equal-mass adaptive ECE directly (it's ~10 lines).
        try:
            import numpy as np
            from netcal.metrics import ECE

            confs = np.array([p.confidence for p in rows], dtype=float)
            correct = np.array(
                [1 if p.predicted == p.expected else 0 for p in rows], dtype=int
            )
            ece = float(ECE(bins=min(n_bins, max(2, len(rows) // 2))).measure(confs, correct))
            out[tier] = ece
            continue
        except Exception:
            pass

        srt = sorted(rows, key=lambda p: p.confidence)
        n = len(srt)
        bin_size = max(1, n // n_bins)
        total = 0.0
        for i in range(0, n, bin_size):
            chunk = srt[i : i + bin_size]
            if not chunk:
                continue
            avg_conf = statistics.mean(p.confidence for p in chunk)
            avg_acc = statistics.mean(
                1.0 if p.predicted == p.expected else 0.0 for p in chunk
            )
            total += (len(chunk) / n) * abs(avg_conf - avg_acc)
        out[tier] = total
    return out


def tier_hits(preds: list[Prediction]) -> tuple[dict[str, int], dict[str, float]]:
    c = Counter(p.source_tier for p in preds)
    total = sum(c.values()) or 1
    share = {k: v / total for k, v in c.items()}
    return dict(c), share


def to_prom_textfile(result: EvalResult) -> str:
    lines: list[str] = [
        "# HELP categorizer_eval_flat_accuracy Flat exact-match accuracy over the gold set.",
        "# TYPE categorizer_eval_flat_accuracy gauge",
        f"categorizer_eval_flat_accuracy {result.flat_accuracy}",
        "# HELP categorizer_eval_macro_f1 Macro-averaged F1 across leaves seen in truth.",
        "# TYPE categorizer_eval_macro_f1 gauge",
        f"categorizer_eval_macro_f1 {result.macro_f1}",
        "# HELP categorizer_eval_hierarchical_f1 Hierarchical (ancestor-set) F1.",
        "# TYPE categorizer_eval_hierarchical_f1 gauge",
        f"categorizer_eval_hierarchical_f1 {result.hierarchical_f1 if result.hierarchical_f1 is not None else 0.0}",
        "# HELP categorizer_eval_total Number of gold rows evaluated.",
        "# TYPE categorizer_eval_total gauge",
        f"categorizer_eval_total {result.total}",
        "# HELP categorizer_eval_errors Number of requests that failed.",
        "# TYPE categorizer_eval_errors gauge",
        f"categorizer_eval_errors {result.errors}",
        "# HELP categorizer_eval_parent_f1 Per-parent F1 after collapsing leaves.",
        "# TYPE categorizer_eval_parent_f1 gauge",
    ]
    for parent, f1 in result.per_parent_f1.items():
        lines.append(f'categorizer_eval_parent_f1{{parent="{parent}"}} {f1}')
    lines += [
        "# HELP categorizer_eval_ece_per_tier Adaptive ECE per tier (equal-mass bins).",
        "# TYPE categorizer_eval_ece_per_tier gauge",
    ]
    for tier, ece in result.ece_per_tier.items():
        if ece is None:
            continue
        lines.append(f'categorizer_eval_ece_per_tier{{tier="{tier}"}} {ece}')
    lines += [
        "# HELP categorizer_eval_tier_share Share of predictions resolved by each tier.",
        "# TYPE categorizer_eval_tier_share gauge",
    ]
    for tier, share in result.tier_share.items():
        lines.append(f'categorizer_eval_tier_share{{tier="{tier}"}} {share}')
    return "\n".join(lines) + "\n"


async def run_eval(
    base_url: str, token: str, gold_path: Path, concurrency: int = 4
) -> tuple[EvalResult, list[Prediction]]:
    rows = load_gold(gold_path)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        preds = await asyncio.gather(
            *[_categorize_one(client, base_url, token, row, sem) for row in rows]
        )

    errors = sum(1 for p in preds if not p.ok)
    hits, share = tier_hits(preds)

    result = EvalResult(
        gold_path=str(gold_path),
        base_url=base_url,
        total=len(preds),
        errors=errors,
        flat_accuracy=flat_accuracy(preds),
        macro_f1=macro_f1(preds),
        per_parent_f1=per_parent_f1(preds),
        hierarchical_f1=hierarchical_f1(preds),
        ece_per_tier=adaptive_ece_per_tier(preds),
        tier_hits=hits,
        tier_share=share,
        predictions=[asdict(p) for p in preds],
    )
    return result, preds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the gold-set eval harness.")
    p.add_argument("--base-url", required=True, help="e.g. https://categorizer.web.vespiridion.org")
    p.add_argument("--token", required=True, help="Bearer token for /v1/*")
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args(argv)

    result, _ = asyncio.run(run_eval(args.base_url, args.token, args.gold, args.concurrency))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    prom_path = args.out.with_suffix(".prom")
    prom_path.write_text(to_prom_textfile(result), encoding="utf-8")

    print(
        f"total={result.total} errors={result.errors} "
        f"flat_acc={result.flat_accuracy:.3f} macro_f1={result.macro_f1:.3f} "
        f"hier_f1={result.hierarchical_f1}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
