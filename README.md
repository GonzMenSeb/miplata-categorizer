# miplata-categorizer

Self-hosted Colombian-Spanish personal-finance transaction categorizer.

Designed to run fully self-hosted on a modest CPU VPS (no GPU, no API fallbacks) and
to *beat* a stateless Claude-Sonnet call on this specific domain by using retrieval
over the user's own labeled history, a Colombian-merchant dictionary, deterministic
rules for internal-transfer detection, and a small tool-using LLM for the tail.

## Cascade

```
raw tx ─► 1. normalize  (Colombian regex: PSE, POS, COMPRA, NEQUI*, DVP, *, etc.)
       ─► 2. feature extract  (temporal, amount bucket, account context, recurring signal)
       ─► 3. rules (deterministic)     ≈ 40% of volume, confidence ≥ 0.95
              • internal transfers between user's own accounts (pair matching)
              • GMF 4x1000 / cuota manejo / mora / intereses
              • Didi / Rappi / EPM / Cívica / Claude / Steam / etc.
       ─► 4. kNN over user's labeled history via pgvector + multilingual-e5-small
              ≈ 30% of volume, confidence ≥ 0.85 with margin ≥ 0.05
       ─► 5. Qwen3-4B-Instruct-2507 /no_think via llama-server + tool use
              ≈ 25% of volume
              tools: lookup_merchant, query_user_history, get_recurring_tx_signal, ask_user_clarification
       ─► 6. Qwen3-4B-Thinking on uncertainty (re-pass with diverse retrieval)
              ≈ 5% of volume
       ─► 7. reject → sin_clasificar.pendiente  (no Claude fallback; user reviews)
```

## Taxonomy

See `config/taxonomy.yaml`. Two-level hierarchy, 13 parents, ~54 children, Spanish.
First-class branch: `movimientos_internos` for movements between the user's own accounts
(Nequi ↔ Bancolombia, pagos de tarjeta propia, retiros de efectivo, recargas de monedero).

Categories marked `implemented: false` are "carcasses" — they exist in the contract
(so user + UI + analytics can reference them) but have no rules or seeded examples yet.
They'll light up as corrections + real data arrive.

## Layout

```
config/taxonomy.yaml       — the category tree
src/categorizer/
  normalize.py             — Colombian-specific text scrubbing
  features.py              — temporal / amount / recurring features
  rules.py                 — deterministic tier
  merchant.py              — merchant resolver (seeded dict; LLM-lookup carcass)
  retrieval.py             — pgvector kNN
  llm.py                   — llama-server OpenAI-compat client, JSON-schema grammar
  tools.py                 — tool implementations
  cascade.py               — orchestrator
  storage.py               — SQLAlchemy models
  api.py                   — HTTP routes
alembic/                   — DB migrations
tests/                     — pytest
```

## Local development

```bash
# From repo root
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'

# Point at a local Postgres + pgvector and a llama-server
cp .env.example .env
uvicorn categorizer.main:app --reload

# Run tests
pytest -v
```

## Deployment

Built by a Jenkins job (`categorizer-deploy`), pushed to the self-hosted Zot
registry as `registry.web.<domain>/miplata/categorizer:latest`, and pulled by the
`categorizer` Ansible role in `vps-infrastructure`. See that repo's
`roles/categorizer/` for the full deploy contract.

## Policy

- **No Claude / third-party API fallback in production.** The `sin_clasificar`
  branch exists for the tail that the local cascade can't resolve confidently;
  the user reviews these manually.
- Confidence scores are calibrated per-tier (see `cascade.py`); tier thresholds
  live in env vars so they're tunable without a redeploy.
