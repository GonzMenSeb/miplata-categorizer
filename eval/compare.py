"""Promotion gate: compare a candidate eval run against a frozen baseline.

Runs two statistical checks against the paired predictions:
  1. McNemar's exact test on the discordant-pair contingency — answers "is
     the candidate statistically different from the baseline on rows where
     only one of them is right?"
  2. Paired bootstrap CI (B=1000) on the macro-F1 delta — answers "how
     confident are we about the direction/size of the improvement?"

Verdict rules (plan §7.3):
  * promote      — macro-F1 delta ≥ +2pp AND McNemar p < 0.05 AND no parent
                   F1 regression > 5pp.
  * reject       — candidate macro-F1 ≥ 2pp WORSE than baseline OR any parent
                   F1 regressed > 5pp.
  * inconclusive — otherwise (insufficient statistical power for a decision).

Exit codes: 0 promote, 1 reject, 2 inconclusive.

First-run pragma: if --baseline does not exist, compare.py prints
"establishing baseline" and exits 0 — the caller is responsible for seeding
eval/baselines/<version>.json from the candidate's metrics.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Verdict:
    outcome: str  # "promote" | "reject" | "inconclusive" | "establishing_baseline"
    reason: str
    macro_f1_baseline: float | None = None
    macro_f1_candidate: float | None = None
    macro_f1_delta: float | None = None
    mcnemar_p: float | None = None
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    worst_parent_regression: tuple[str, float] | None = None


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mcnemar_p(baseline: list[bool], candidate: list[bool]) -> float:
    """Exact McNemar's test on paired correctness bitstrings.

    Uses the two-sided binomial tail on the smaller discordant count; this
    matches the "exact" variant and does not require the >25-discordant
    normal approximation.
    """
    assert len(baseline) == len(candidate)
    b_only = sum(1 for b, c in zip(baseline, candidate, strict=True) if b and not c)
    c_only = sum(1 for b, c in zip(baseline, candidate, strict=True) if c and not b)
    n = b_only + c_only
    if n == 0:
        return 1.0
    k = min(b_only, c_only)
    # P(X <= k | X~Bin(n, 0.5)) * 2, clipped to [0,1].
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _f1_score(pairs: list[tuple[str, str]]) -> float:
    classes = {t for t, _ in pairs}
    if not classes:
        return 0.0
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in pairs if t == cls and p == cls)
        fp = sum(1 for t, p in pairs if t != cls and p == cls)
        fn = sum(1 for t, p in pairs if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def paired_bootstrap_ci(
    baseline_pairs: list[tuple[str, str]],
    candidate_pairs: list[tuple[str, str]],
    b: int = 1000,
    alpha: float = 0.05,
    seed: int = 0xC0DE,
) -> tuple[float, float]:
    """Paired bootstrap CI on macro-F1 delta (candidate - baseline).

    Returns (low, high) of a two-sided (1-alpha) interval.
    """
    assert len(baseline_pairs) == len(candidate_pairs)
    n = len(baseline_pairs)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        bs_b = [baseline_pairs[i] for i in idx]
        bs_c = [candidate_pairs[i] for i in idx]
        deltas.append(_f1_score(bs_c) - _f1_score(bs_b))
    deltas.sort()
    lo = deltas[max(0, int((alpha / 2) * b) - 1)]
    hi = deltas[min(b - 1, int((1 - alpha / 2) * b) - 1)]
    return lo, hi


def _pairs(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in metrics.get("predictions", []):
        if not p.get("ok"):
            continue
        out.append((p["expected"], p["predicted"]))
    return out


def decide(baseline: dict[str, Any], candidate: dict[str, Any]) -> Verdict:
    b_pairs_full = _pairs(baseline)
    c_pairs_full = _pairs(candidate)
    b_by_id = {p["external_id"]: p for p in baseline.get("predictions", []) if p.get("ok")}
    c_by_id = {p["external_id"]: p for p in candidate.get("predictions", []) if p.get("ok")}
    common_ids = sorted(set(b_by_id) & set(c_by_id))
    if not common_ids:
        return Verdict(
            outcome="inconclusive",
            reason="No shared external_ids between baseline and candidate predictions.",
        )

    b_pairs = [(b_by_id[i]["expected"], b_by_id[i]["predicted"]) for i in common_ids]
    c_pairs = [(c_by_id[i]["expected"], c_by_id[i]["predicted"]) for i in common_ids]
    b_correct = [b_by_id[i]["predicted"] == b_by_id[i]["expected"] for i in common_ids]
    c_correct = [c_by_id[i]["predicted"] == c_by_id[i]["expected"] for i in common_ids]

    macro_b = float(baseline.get("macro_f1", _f1_score(b_pairs_full and b_pairs)))
    macro_c = float(candidate.get("macro_f1", _f1_score(c_pairs_full and c_pairs)))
    delta = macro_c - macro_b
    p = mcnemar_p(b_correct, c_correct)
    ci_lo, ci_hi = paired_bootstrap_ci(b_pairs, c_pairs)

    # Parent regression check.
    pp_b = baseline.get("per_parent_f1", {})
    pp_c = candidate.get("per_parent_f1", {})
    worst: tuple[str, float] | None = None
    for parent, b_f1 in pp_b.items():
        c_f1 = pp_c.get(parent, 0.0)
        drop = float(b_f1) - float(c_f1)
        if worst is None or drop > worst[1]:
            worst = (parent, drop)

    common = {
        "macro_f1_baseline": macro_b,
        "macro_f1_candidate": macro_c,
        "macro_f1_delta": delta,
        "mcnemar_p": p,
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
        "worst_parent_regression": worst,
    }

    # Clear reject: ≥2pp worse or parent regression >5pp.
    if delta <= -0.02 or (worst is not None and worst[1] > 0.05):
        return Verdict(
            outcome="reject",
            reason=f"delta={delta:+.3f} worst_parent_drop={worst}",
            **common,
        )
    # Clear promote: +2pp better AND p<0.05 AND no parent >5pp drop.
    if delta >= 0.02 and p < 0.05 and (worst is None or worst[1] <= 0.05):
        return Verdict(
            outcome="promote",
            reason=f"delta={delta:+.3f} p={p:.3f} worst_parent_drop={worst}",
            **common,
        )
    return Verdict(
        outcome="inconclusive",
        reason=(
            f"delta={delta:+.3f} p={p:.3f} ci=[{ci_lo:+.3f},{ci_hi:+.3f}] "
            f"worst_parent_drop={worst} — not enough evidence to decide."
        ),
        **common,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare a candidate eval run to a baseline.")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.baseline.exists():
        verdict = Verdict(
            outcome="establishing_baseline",
            reason=(
                f"{args.baseline} does not exist — this is the first run. "
                "Seed it by copying the candidate metrics.json into that path "
                "and commit it as eval/baselines/v0.json."
            ),
        )
        print(json.dumps(asdict(verdict), indent=2))
        return 0

    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)
    verdict = decide(baseline, candidate)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(asdict(verdict), indent=2), encoding="utf-8")
    print(json.dumps(asdict(verdict), indent=2))

    if verdict.outcome == "promote":
        return 0
    if verdict.outcome == "reject":
        return 1
    if verdict.outcome == "establishing_baseline":
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
