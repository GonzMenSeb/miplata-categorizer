# Fine-tune pipeline

QLoRA fine-tune of Qwen3-4B-Instruct-2507 on Sebas's labeled Colombian-Spanish transaction corpus.

## When to run

Three triggers should all be satisfied:
1. **Corpus ≥ 500 labeled rows** in `labeled_transactions` (excluding `sin_clasificar.pendiente`). Current corpus: check via `SELECT count(*) FROM labeled_transactions WHERE category_slug <> 'sin_clasificar.pendiente';`.
2. **Macro-F1 on `gold_set_v1` has plateaued** for 2+ eval runs. If rules + merchant + kNN keep pulling it up, fine-tune gains less.
3. **LLM-path share > 20%** of production traffic. If the LLM tier rarely fires, tuning it has no impact.

## Hardware

**DO NOT run on the VPS.** Haswell CPU cannot host Unsloth — it needs CUDA.

Target: RunPod Community Cloud RTX 4090.
- ~$0.34/hr
- ~3h per full 3-epoch run
- ~$1.50 total per tune

Storage on the pod: ~30GB (base model + dataset + artifacts).

## Runbook

### 1. Export the corpus (runs on VPS, inside the categorizer-api container)

```bash
TOKEN=$(ANSIBLE_VAULT_PASSWORD_FILE=/path/to/.vault_pass \
  ansible-vault view /path/to/vault.yml \
  | awk -F'"' '/vault_categorizer_api_token/ {print $2; exit}')

ansible vps -e "@/path/to/vault.yml" -i /path/to/inventory.yml -m shell \
  -a 'docker exec categorizer-api python /app/scripts/export_ft_corpus.py \
      --db-url "$DATABASE_URL" --out /tmp/corpus/'

# Fetch corpus locally
scp -r ubuntu@<vps>:/tmp/corpus/ ./training/corpus/
```

The script warns if corpus < 500 but does not refuse. Proceed anyway only if you accept a weaker training signal.

### 2. Spin up a RunPod pod

- Image: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` (or current)
- GPU: 1x RTX 4090 (24GB VRAM)
- Disk: 50GB
- Open port 22 for scp.

### 3. Install training deps on the pod

```bash
git clone git@github.com:GonzMenSeb/miplata-categorizer.git
cd miplata-categorizer
pip install -e '.[training]'
```

### 4. Run the tune

```bash
python -m training.train --run \
    --corpus training/corpus/ \
    --out training/artifacts/run/
```

Dry-run first (without `--run`) to verify arguments print correctly.

### 5. Evaluate locally on the pod

```bash
python -m training.eval_ft --run \
    --merged training/artifacts/run/merged \
    --eval-set training/corpus/eval.jsonl \
    --out training/artifacts/run/eval_metrics.json
```

### 6. Convert merged weights → GGUF

This uses upstream llama.cpp tooling, not miplata-categorizer code:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make
python convert_hf_to_gguf.py ../miplata-categorizer/training/artifacts/run/merged \
    --outtype bf16 --outfile qwen3-4b-miplata-v1.bf16.gguf
./llama-quantize qwen3-4b-miplata-v1.bf16.gguf qwen3-4b-miplata-v1.gguf Q4_K_M
```

### 7. Deploy to VPS

```bash
scp qwen3-4b-miplata-v1.gguf ubuntu@<vps>:/opt/llm_inference/models/
```

Then on the operator laptop, in `vps-infrastructure/`:

```bash
# Edit roles/llm_inference/defaults/main.yml:
#   llm_inference_model_file: "qwen3-4b-miplata-v1.gguf"
ANSIBLE_VAULT_PASSWORD_FILE=.vault_pass ansible-playbook playbook.yml
```

This re-renders `/opt/llm_inference/.env` and recreates the llama-server container with the new model.

### 8. A/B verify before promotion

```bash
# Against gold set
TOKEN=... PYTHONPATH=src python -m eval.eval \
    --base-url https://categorizer.web.vespiridion.org \
    --token $TOKEN \
    --gold config/gold_set_v1.jsonl \
    --out eval/metrics-post-tune.json

python -m eval.compare \
    --baseline eval/baselines/v0.json \
    --candidate eval/metrics-post-tune.json
```

Promotion criteria:
- macro-F1 improved ≥ 2pp vs baseline
- No parent F1 dropped > 5pp
- McNemar p < 0.05 (compare.py enforces this)
- Optional: MMLU-100 canary didn't drop > 3pp (run a tiny MMLU-Es slice if paranoid)

### 9. If A/B fails → rollback

```bash
# Edit roles/llm_inference/defaults/main.yml back to the prior gguf
ansible-playbook playbook.yml
# Or keep both files on disk and flip via env override
```

## Cadence

- While label stream is heavy (weeks after onboarding the correction UI): **weekly retrain**.
- Steady-state: **monthly**.
- Always: re-baseline against Claude Sonnet every 3 months (plan.md §7.1 hard constraint).

## Related files

- `training/dataset.py` — corpus export
- `training/train.py` — Unsloth tuner (gated behind `--run`)
- `training/eval_ft.py` — post-tune eval (gated)
- `scripts/export_ft_corpus.py` — thin CLI wrapper around `training.dataset`
- `eval/compare.py` — promotion gate
- `eval/baselines/v0.json` — current baseline; do not edit by hand
