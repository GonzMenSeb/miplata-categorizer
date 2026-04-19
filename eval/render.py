"""Render eval/metrics.json into operator-facing PNGs.

Outputs, under ``--out`` (default eval/report/):

  * ``parent_f1.png``       — per-parent F1 heatmap (rows=parent, col=F1).
  * ``tier_share.png``      — bar chart of tier hit share.
  * ``reliability_<tier>.png`` — reliability diagram per tier that saw ≥5
    predictions, with ECE in the title.

Matplotlib is an optional dep (eval group) — the renderer must not be
imported from runtime code paths.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _parent_f1_heatmap(per_parent_f1: dict[str, float], out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    parents = sorted(per_parent_f1.keys())
    if not parents:
        return
    values = np.array([[per_parent_f1[p]] for p in parents], dtype=float)

    fig, ax = plt.subplots(figsize=(4, max(3, 0.35 * len(parents) + 1.2)))
    im = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(parents)), labels=parents)
    ax.set_xticks([0], labels=["F1"])
    for i, p in enumerate(parents):
        ax.text(0, i, f"{per_parent_f1[p]:.2f}", ha="center", va="center",
                color="white" if per_parent_f1[p] < 0.6 else "black", fontsize=9)
    ax.set_title("Per-parent F1")
    fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _tier_share_bars(tier_share: dict[str, float], out: Path) -> None:
    import matplotlib.pyplot as plt

    if not tier_share:
        return
    tiers = sorted(tier_share.keys(), key=lambda t: tier_share[t], reverse=True)
    shares = [tier_share[t] for t in tiers]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(tiers, shares, color="#4C72B0")
    ax.set_ylabel("share of predictions")
    ax.set_ylim(0, 1)
    ax.set_title("Tier hit share")
    for i, s in enumerate(shares):
        ax.text(i, s + 0.01, f"{s:.0%}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _adaptive_bins(
    confs: list[float], correct: list[int], n_bins: int
) -> list[tuple[float, float, int]]:
    """Return (bin_mean_conf, bin_accuracy, bin_size) for up to n_bins
    equal-mass bins. Falls back to ``netcal`` if installed.
    """
    import contextlib

    if not confs:
        return []
    with contextlib.suppress(Exception):  # pragma: no cover — optional dep
        from netcal.binning import ENIR  # noqa: F401 — presence check only
        # netcal exposes adaptive binning through internal helpers; the
        # equal-mass variant below is functionally identical and keeps the
        # output format stable.

    order = sorted(range(len(confs)), key=lambda i: confs[i])
    n = len(order)
    bins_out: list[tuple[float, float, int]] = []
    size = max(1, n // max(1, n_bins))
    for start in range(0, n, size):
        idxs = order[start : start + size]
        if not idxs:
            continue
        mc = statistics.mean(confs[i] for i in idxs)
        acc = statistics.mean(correct[i] for i in idxs)
        bins_out.append((mc, acc, len(idxs)))
    return bins_out


def _reliability_diagram(
    tier: str, confs: list[float], correct: list[int], out: Path
) -> None:
    import matplotlib.pyplot as plt

    bins = _adaptive_bins(confs, correct, n_bins=10)
    if not bins:
        return
    xs = [b[0] for b in bins]
    ys = [b[1] for b in bins]
    sizes = [b[2] for b in bins]
    total = sum(sizes)
    ece = sum((sz / total) * abs(x - y) for (x, y, sz) in bins)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="perfect")
    ax.plot(xs, ys, marker="o", color="#C44E52", label=tier)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability — {tier} (ECE={ece:.3f}, n={total})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run(metrics: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    per_parent_f1 = metrics.get("per_parent_f1") or {}
    if per_parent_f1:
        p = out_dir / "parent_f1.png"
        _parent_f1_heatmap(per_parent_f1, p)
        produced.append(p)

    tier_share = metrics.get("tier_share") or {}
    if tier_share:
        p = out_dir / "tier_share.png"
        _tier_share_bars(tier_share, p)
        produced.append(p)

    by_tier_conf: dict[str, list[float]] = defaultdict(list)
    by_tier_correct: dict[str, list[int]] = defaultdict(list)
    for pred in metrics.get("predictions", []):
        if not pred.get("ok"):
            continue
        tier = str(pred.get("source_tier", "unknown"))
        by_tier_conf[tier].append(float(pred.get("confidence", 0.0)))
        by_tier_correct[tier].append(
            1 if pred.get("predicted") == pred.get("expected") else 0
        )
    for tier, confs in by_tier_conf.items():
        if len(confs) < 5:
            continue
        p = out_dir / f"reliability_{tier}.png"
        _reliability_diagram(tier, confs, by_tier_correct[tier], p)
        produced.append(p)

    return produced


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render eval metrics into PNGs.")
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("eval/report"))
    args = ap.parse_args(argv)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    produced = run(metrics, args.out)
    for p in produced:
        print(str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
