from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.compare import (  # noqa: E402
    decide,
    main,
    mcnemar_p,
    paired_bootstrap_ci,
)


def test_mcnemar_identical_predictions_is_one() -> None:
    assert mcnemar_p([True, False, True], [True, False, True]) == 1.0


def test_mcnemar_one_sided_flip_gives_small_p() -> None:
    # 10 rows where candidate is always right, baseline always wrong.
    p = mcnemar_p([False] * 10, [True] * 10)
    assert p < 0.05


def test_paired_bootstrap_ci_sensible() -> None:
    b = [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")] * 5
    c = [("a", "a"), ("a", "a"), ("b", "b"), ("b", "b")] * 5
    lo, hi = paired_bootstrap_ci(b, c, b=200, seed=1)
    assert lo <= hi


def _metrics(predictions: list[tuple[str, str, str]], parent_f1: dict[str, float], macro: float) -> dict:
    return {
        "macro_f1": macro,
        "per_parent_f1": parent_f1,
        "predictions": [
            {"external_id": eid, "expected": t, "predicted": p, "ok": True}
            for eid, t, p in predictions
        ],
    }


def test_decide_promote_when_candidate_wins_big() -> None:
    # Candidate corrects every baseline miss on a 12-row set.
    b_preds = [(f"r{i}", "a.x", "a.y") for i in range(12)]
    c_preds = [(f"r{i}", "a.x", "a.x") for i in range(12)]
    baseline = _metrics(b_preds, {"a": 0.0}, 0.0)
    candidate = _metrics(c_preds, {"a": 1.0}, 1.0)
    v = decide(baseline, candidate)
    assert v.outcome == "promote"


def test_decide_reject_when_candidate_regresses() -> None:
    b_preds = [(f"r{i}", "a.x", "a.x") for i in range(10)]
    c_preds = [(f"r{i}", "a.x", "a.y") for i in range(10)]
    baseline = _metrics(b_preds, {"a": 1.0}, 1.0)
    candidate = _metrics(c_preds, {"a": 0.0}, 0.0)
    v = decide(baseline, candidate)
    assert v.outcome == "reject"


def test_decide_inconclusive_when_tiny() -> None:
    b_preds = [(f"r{i}", "a.x", "a.x") for i in range(3)]
    c_preds = [(f"r{i}", "a.x", "a.x") for i in range(3)]
    baseline = _metrics(b_preds, {"a": 1.0}, 1.0)
    candidate = _metrics(c_preds, {"a": 1.0}, 1.0)
    v = decide(baseline, candidate)
    assert v.outcome == "inconclusive"


def test_main_missing_baseline_exits_zero(tmp_path: Path, capsys) -> None:
    candidate = tmp_path / "metrics.json"
    candidate.write_text(json.dumps(_metrics([], {}, 0.0)))
    rc = main(["--baseline", str(tmp_path / "nope.json"), "--candidate", str(candidate)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "establishing_baseline" in out
