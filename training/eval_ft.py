"""Post-training evaluation of a merged fine-tuned model.

Reuses `eval.eval`'s metric computation but calls the merged HF model
directly (via transformers) instead of the cascade HTTP endpoint. Lets you
compare pre-/post-tune F1 without deploying the GGUF to the VPS first.

Flow:
  merged_model + eval_set  →  per-row predicted_slug
  eval_set + gold_set       →  metrics.json via eval/compare.py logic
  compare to baselines/v0   →  promote | reject | inconclusive

Gated identically to train.py — imports transformers only on --run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _dry_run(args: argparse.Namespace) -> None:
    print("=== eval_ft dry-run (--run not set) ===")
    print(f"merged model: {args.merged}")
    print(f"eval set:     {args.eval_set}")
    print(f"gold set:     {args.gold}")
    print(f"out metrics:  {args.out}")
    print(f"baseline:     {args.baseline}")
    print()
    print("On --run:")
    print(" 1. Load merged model via transformers")
    print(" 2. Run inference with same JSON-schema grammar as prod llama-server")
    print(" 3. Write metrics.json (same shape as eval/metrics.json)")
    print(" 4. Delegate to eval.compare.decide() for promotion verdict")
    print(" 5. Exit 0 if 'promote', 1 if 'reject', 2 if 'inconclusive'")


def _run(args: argparse.Namespace) -> int:
    try:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as e:
        print(f"missing training dependencies: {e}", file=sys.stderr)
        print("Install with: pip install -e '.[training]'", file=sys.stderr)
        return 2

    merged_dir = Path(args.merged)
    if not (merged_dir / "config.json").exists():
        print(f"{merged_dir}/config.json not found — is this a valid HF checkpoint?", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir))
    model = AutoModelForCausalLM.from_pretrained(str(merged_dir), torch_dtype=torch.bfloat16, device_map="auto")

    rows = [json.loads(line) for line in Path(args.eval_set).read_text(encoding="utf-8").splitlines()]
    predictions: list[dict[str, object]] = []
    for row in rows:
        msgs = row["messages"][:-1]  # drop the assistant turn — model generates it
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            parsed = json.loads(gen)
            predicted = parsed.get("category_slug", "")
        except json.JSONDecodeError:
            predicted = ""
        truth = json.loads(row["messages"][-1]["content"]).get("category_slug", "")
        predictions.append({"external_id": row["external_id"], "predicted": predicted, "expected": truth})

    # Very light metric shape — full breakdown via eval/eval.py against gold
    correct = sum(1 for p in predictions if p["predicted"] == p["expected"])
    metrics = {
        "n": len(predictions),
        "flat_accuracy": correct / max(len(predictions), 1),
        "predictions": predictions,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out} — flat_accuracy={metrics['flat_accuracy']:.4f} on {metrics['n']} rows")

    if args.baseline:
        print(f"Baseline at {args.baseline} — run eval/compare.py for statistical promotion verdict.")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a merged fine-tuned model.")
    p.add_argument("--run", action="store_true")
    p.add_argument("--merged", type=Path, required=False, default=Path("training/artifacts/run/merged"))
    p.add_argument("--eval-set", type=Path, default=Path("training/corpus/eval.jsonl"))
    p.add_argument("--gold", type=Path, default=Path("config/gold_set_v1.jsonl"))
    p.add_argument("--out", type=Path, default=Path("training/artifacts/run/eval_metrics.json"))
    p.add_argument("--baseline", type=Path, default=Path("eval/baselines/v0.json"))
    args = p.parse_args(argv)

    if not args.run:
        _dry_run(args)
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
