from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from eval.eval import (  # noqa: E402
    EvalResult,
    Prediction,
    adaptive_ece_per_tier,
    flat_accuracy,
    hierarchical_f1,
    macro_f1,
    per_parent_f1,
    tier_hits,
    to_prom_textfile,
)


def _p(expected: str, predicted: str, tier: str = "rules", conf: float = 0.9) -> Prediction:
    return Prediction(
        external_id="x",
        expected=expected,
        predicted=predicted,
        confidence=conf,
        source_tier=tier,
        latency_ms=1.0,
        ok=True,
    )


def test_flat_accuracy_all_correct() -> None:
    preds = [_p("comida.restaurantes", "comida.restaurantes"), _p("ocio.cine_teatro", "ocio.cine_teatro")]
    assert flat_accuracy(preds) == 1.0


def test_flat_accuracy_mixed() -> None:
    preds = [_p("comida.restaurantes", "comida.restaurantes"), _p("ocio.cine_teatro", "comida.restaurantes")]
    assert flat_accuracy(preds) == 0.5


def test_macro_f1_handles_partial_miss() -> None:
    preds = [
        _p("comida.restaurantes", "comida.restaurantes"),
        _p("comida.restaurantes", "comida.restaurantes"),
        _p("ocio.cine_teatro", "comida.restaurantes"),
    ]
    assert 0.0 < macro_f1(preds) < 1.0


def test_per_parent_f1_collapses_leaves() -> None:
    preds = [
        _p("comida.restaurantes", "comida.mercado"),  # parent right, leaf wrong
        _p("comida.mercado", "comida.mercado"),
    ]
    parents = per_parent_f1(preds)
    assert parents["comida"] == 1.0


def test_hierarchical_f1_partial_credit() -> None:
    preds = [_p("comida.restaurantes", "comida.mercado")]
    hf = hierarchical_f1(preds)
    assert hf is not None
    assert 0.0 < hf < 1.0


def test_tier_hits_share_sums_to_one() -> None:
    preds = [
        _p("a.b", "a.b", tier="rules"),
        _p("a.b", "a.b", tier="knn"),
        _p("a.b", "a.b", tier="llm_notink"),
    ]
    hits, share = tier_hits(preds)
    assert sum(hits.values()) == 3
    assert abs(sum(share.values()) - 1.0) < 1e-9


def test_adaptive_ece_skips_small_tiers() -> None:
    preds = [_p("a.b", "a.b", tier="rules", conf=0.9)]
    out = adaptive_ece_per_tier(preds)
    assert out["rules"] is None


def test_prom_textfile_has_core_metrics() -> None:
    preds = [_p("a.b", "a.b", tier="rules")]
    hits, share = tier_hits(preds)
    result = EvalResult(
        gold_path="x",
        base_url="y",
        total=1,
        errors=0,
        flat_accuracy=1.0,
        macro_f1=1.0,
        per_parent_f1={"a": 1.0},
        hierarchical_f1=1.0,
        ece_per_tier={"rules": 0.0},
        tier_hits=hits,
        tier_share=share,
    )
    body = to_prom_textfile(result)
    assert "categorizer_eval_flat_accuracy 1.0" in body
    assert "categorizer_eval_parent_f1" in body
    assert 'tier="rules"' in body
