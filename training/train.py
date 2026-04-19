"""QLoRA fine-tune of Qwen3-4B-Instruct-2507 on Sebas's labeled corpus.

DO NOT run on the VPS. Haswell CPU cannot host Unsloth; use a RunPod pod
with an RTX 4090 (Community Cloud, ~$0.34/hr). See training/README.md.

Recipe (plan.md §7.1):
  r=16, lora_alpha=32, target_modules="all-linear",
  lr=2e-4, cosine scheduler, 3 epochs, bf16
  10% Tulu-3 mixin for instruction preservation (optional)

Gated: without `--run`, this script validates arguments + prints what it
would do, then exits. No Unsloth/PEFT/TRL import happens unless --run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _dry_run(args: argparse.Namespace) -> None:
    print("=== fine-tune dry-run (--run not set) ===")
    print(f"base model:       {args.base_model}")
    print(f"corpus dir:       {args.corpus}")
    print(f"output dir:       {args.out}")
    print(f"LoRA rank r:      {args.lora_r}")
    print(f"LoRA alpha:       {args.lora_alpha}")
    print(f"target modules:   {args.target_modules}")
    print(f"lr:               {args.lr}")
    print(f"epochs:           {args.epochs}")
    print(f"batch size:       {args.batch_size}")
    print(f"grad accum:       {args.grad_accum}")
    print(f"bf16:             {args.bf16}")
    print(f"tulu3 mixin:      {args.tulu3_mixin_frac}")
    print()
    print("To actually train, pass --run. Expected runtime on RTX 4090: ~3h.")
    print("Recommended next step (post-train):")
    print("  1. python -m training.eval_ft --merged <out>/merged --baseline eval/baselines/v0.json")
    print("  2. upstream: llama.cpp/convert_hf_to_gguf.py <merged> → bf16.gguf")
    print("  3. upstream: llama-quantize bf16.gguf qwen3-4b-miplata-v1.gguf Q4_K_M")
    print("  4. scp qwen3-4b-miplata-v1.gguf to VPS /opt/llm_inference/models/")
    print("  5. update llm_inference_model_file default + ansible-playbook")


def _train(args: argparse.Namespace) -> None:
    # These imports are intentionally deferred — the categorizer runtime
    # image does NOT ship unsloth / peft / trl (they'd bloat the image by
    # ~4GB). This script is only runnable from a training environment.
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]
    except ImportError as e:
        print(f"training dependencies not installed: {e}", file=sys.stderr)
        print("Install with: pip install -e '.[training]'", file=sys.stderr)
        sys.exit(2)

    corpus = Path(args.corpus)
    train_path = corpus / "train.jsonl"
    eval_path = corpus / "eval.jsonl"
    if not train_path.exists():
        print(f"missing {train_path} — run `python -m training.dataset --out {corpus}` first", file=sys.stderr)
        sys.exit(2)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect; bf16 on Ampere+
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("json", data_files={"train": str(train_path), "eval": str(eval_path)})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="messages",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=args.bf16,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            seed=args.seed,
            report_to="none",
        ),
    )
    trainer.train()

    merged_dir = Path(args.out) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    # Unsloth's merge-to-16bit writes a standalone HF checkpoint ready for GGUF
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"merged model saved → {merged_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QLoRA fine-tune Qwen3-4B on Sebas's corpus.")
    p.add_argument("--run", action="store_true", help="Actually train. Without this flag, print config and exit.")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--corpus", type=Path, default=Path("training/corpus"), help="Dir containing train.jsonl + eval.jsonl")
    p.add_argument("--out", type=Path, default=Path("training/artifacts/run"))
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--target-modules", default="all-linear")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--tulu3-mixin-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not args.run:
        _dry_run(args)
        return 0
    _train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
