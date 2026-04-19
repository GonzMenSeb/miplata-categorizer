"""Fine-tune pipeline for Qwen3-4B on Sebas's labeled Colombian-Spanish corpus.

See `training/README.md` for the full runbook. Trigger condition:
corpus >= 500 labeled rows AND macro-F1 plateaued on gold_set_v1.

Modules:
  - dataset.py   — pull labeled_transactions → ChatML JSONL, 90/10 split.
  - train.py     — Unsloth + PEFT + TRL QLoRA; gated behind --run.
  - eval_ft.py   — evaluate a merged model against hold-out + gold_set_v1.
"""
