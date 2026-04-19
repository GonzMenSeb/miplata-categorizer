"""Export labeled_transactions as Qwen3 ChatML JSONL for QLoRA fine-tuning.

The prompt shape mirrors cascade._build_llm_messages at inference time so
the tuned model sees the same task during training and prod. We deliberately
do NOT depend on cascade.py at import time (cascade pulls in sqlalchemy/
llama-client graphs that RunPod's training image doesn't need) — we rebuild
the prompt skeleton inline and keep it in sync by shape alone.

Skipped rows:
  - `category_slug == 'sin_clasificar.pendiente'` (these are "we don't know"
    rejects — training on them teaches the model to reject, not useful).
  - Rows with NULL embedding (should never happen post-seed, but defensive).

Split: 90/10 train/eval, seeded RNG for reproducibility across runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text

_SYSTEM_PROMPT = (
    "Eres un categorizador de transacciones bancarias colombianas para una app "
    "de finanzas personales. Elige EXACTAMENTE un slug del enum provisto. "
    "Si la transacción parece moverse entre cuentas propias del usuario, "
    "usa la rama `movimientos_internos`. Si la transacción es genuinamente "
    "ambigua y ninguna herramienta ayuda, usa `sin_clasificar.pendiente` — "
    "NO adivines."
)

REJECT_SLUG = "sin_clasificar.pendiente"
MIN_CORPUS_WARN = 500


@dataclass(frozen=True)
class TrainingRow:
    external_id: str
    messages: list[dict[str, str]]  # ChatML: system, user, assistant

    def to_json(self) -> str:
        return json.dumps({"external_id": self.external_id, "messages": self.messages}, ensure_ascii=False)


def _build_user_message(
    description: str,
    normalized: str,
    amount: float,
    currency: str,
    transaction_type: str,
    tx_date: str,
    account_slug: str,
) -> str:
    # Neighbors / own-accounts / taxonomy are deliberately OMITTED from the
    # training prompt — we train the model to classify from tx fields only.
    # At inference the cascade provides them as hints, but the training
    # signal is "given the raw tx, what's the correct slug".
    return (
        f"Transacción:\n"
        f"  descripción original: {description!r}\n"
        f"  descripción normalizada: {normalized!r}\n"
        f"  monto: {amount} {currency}\n"
        f"  tipo: {transaction_type}\n"
        f"  fecha: {tx_date}\n"
        f"  cuenta: {account_slug}\n\n"
        "Responde SÓLO con el JSON que cumple el schema."
    )


def _build_assistant_message(category_slug: str, confidence: float, reasoning: str) -> str:
    return json.dumps(
        {"category_slug": category_slug, "confidence": confidence, "reasoning": reasoning},
        ensure_ascii=False,
    )


def fetch_rows(db_url: str) -> list[TrainingRow]:
    """Fetch non-reject labeled rows, prioritizing user corrections.

    Corrections carry higher signal than seeded labels (they reflect what the
    cascade got wrong in production). We pull corrections first, then fill
    with seed labels.
    """
    # psycopg v3 handles sync + async via the same driver; the runtime
    # container only has psycopg v3 installed, so leave the URL scheme alone.
    engine = create_engine(db_url)
    rows: list[TrainingRow] = []
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT
                    lt.external_id, lt.tx_date, lt.amount, lt.currency,
                    lt.transaction_type, lt.description, lt.normalized_description,
                    lt.account_slug, lt.category_slug, lt.source
                FROM labeled_transactions lt
                WHERE lt.embedding IS NOT NULL
                  AND lt.category_slug <> :reject_slug
                ORDER BY (CASE WHEN lt.source = 'user' THEN 0 ELSE 1 END), lt.tx_date DESC
                """
            ),
            {"reject_slug": REJECT_SLUG},
        )
        for r in result:
            user_msg = _build_user_message(
                description=r.description,
                normalized=r.normalized_description,
                amount=float(r.amount),
                currency=r.currency,
                transaction_type=r.transaction_type,
                tx_date=r.tx_date.isoformat(),
                account_slug=r.account_slug,
            )
            # A generic reasoning — the model learns the mapping, not the prose
            assistant_msg = _build_assistant_message(
                category_slug=r.category_slug,
                confidence=0.95 if r.source == "user" else 0.90,
                reasoning=f"Categoría validada por {'usuario' if r.source == 'user' else 'etiqueta seed'}.",
            )
            rows.append(
                TrainingRow(
                    external_id=r.external_id,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ],
                )
            )
    return rows


def split_rows(rows: list[TrainingRow], eval_frac: float = 0.10, seed: int = 42) -> tuple[list[TrainingRow], list[TrainingRow]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * eval_frac))
    return shuffled[cut:], shuffled[:cut]


def export(db_url: str, out_dir: Path, min_rows: int = MIN_CORPUS_WARN) -> None:
    rows = fetch_rows(db_url)
    if not rows:
        print("No labeled rows found. Nothing to export.", file=sys.stderr)
        return

    if len(rows) < min_rows:
        print(
            f"WARNING: corpus size {len(rows)} < {min_rows}. Fine-tuning on this "
            "is UNLIKELY to measurably beat zero-shot — per plan.md §7.1, wait "
            "for ~500+ labeled rows. Proceeding anyway (requested).",
            file=sys.stderr,
        )

    train, evalset = split_rows(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train:
            f.write(r.to_json() + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for r in evalset:
            f.write(r.to_json() + "\n")

    print(f"wrote {len(train)} train rows → {train_path}")
    print(f"wrote {len(evalset)} eval rows  → {eval_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export labeled_transactions as ChatML JSONL.")
    p.add_argument("--db-url", required=True, help="postgresql+psycopg:// DSN")
    p.add_argument("--out", required=True, type=Path, help="Output directory for train.jsonl + eval.jsonl")
    p.add_argument("--min-rows", type=int, default=MIN_CORPUS_WARN, help="Warn if corpus < this threshold")
    args = p.parse_args(argv)
    export(args.db_url, args.out, min_rows=args.min_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
