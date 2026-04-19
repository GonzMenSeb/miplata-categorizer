# plan.md — technical plan, status, and roadmap

> Purpose: implementation-grade specification of how we're building the
> miplata-categorizer, what's done, what's pending, and exactly how to do the
> remaining pieces. For the "why", read `context.md`.

---

## 1. Target architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                               miplata-api                                    │
│  (NestJS — reads CATEGORIZER_BASE_URL + CATEGORIZER_API_TOKEN from env)     │
└────────────────────┬─────────────────────────────────────────────────────────┘
                     │ HTTPS + Bearer
                     ▼
              Traefik (main VPS)
           categorizer.web.<domain>
             apitokenmiddleware
                     │
                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         categorizer-api (FastAPI)                            │
│                                                                              │
│  Tier 1. normalize()         — Colombian-Spanish regex (see src/normalize.py)│
│  Tier 2. features            — temporal, amount bucket, recurring signal     │
│  Tier 3. rules.match()       — deterministic regex + own_accounts cross-ref  │
│  Tier 4. paired-tx check     — DB scan for opposite-sign same-amount tx      │
│  Tier 5. merchant lookup     — SQLite-like exact/substring alias match       │
│  Tier 6. pgvector kNN        — fastembed multilingual-e5-small, top-k=8      │
│  Tier 7. LLM /no_think       — Qwen3-4B via llama-server (Hermes tool call)  │
│  Tier 8. LLM /think          — Qwen3-4B-Thinking re-pass on low confidence   │
│  Tier 9. reject              — sin_clasificar.pendiente (no API fallback)    │
│                                                                              │
│  tools:  lookup_merchant, query_user_history, get_recurring_tx_signal,       │
│          ask_user_clarification                                              │
└─────────┬───────────────┬───────────────┬────────────────────────────────────┘
          │               │               │
          ▼               ▼               ▼
  categorizer-         llama-server    miplata-postgres (read-only)
  postgres                (Qwen3-4B)   via MIPLATA_RO_DATABASE_URL
  (pgvector)
     │
  own_accounts, labeled_transactions (vector(384)),
  predictions, corrections, merchants
```

Tier split targets, by expected traffic share:

| Tier | Target share | Latency | Confidence floor |
|---|---|---|---|
| 1. rules (text + paired-tx) | ~40% | < 10 ms | 0.95 |
| 2. kNN | ~30% | 30–100 ms | 0.85 w/ margin ≥ 0.05 |
| 3. LLM /no_think | ~25% | 7–10 s | 0.70 |
| 4. LLM /think | ~5% | 15–25 s | 0.70 |
| 5. reject → pendiente | residual | — | n/a |

## 2. Stack decisions

| Layer | Choice | Why (short) |
|---|---|---|
| Language / runtime | Python 3.12 | Matches Qwen3 / llama.cpp ecosystem defaults; Node 24 fetch used only on the miplata client side |
| Web framework | FastAPI + Uvicorn (1 worker, threadpool) | Async, OpenAPI for free, trivial bearer-guard via Traefik middleware upstream |
| DB ORM | SQLAlchemy 2.0 async + psycopg (v3, binary) | First-class async, clean typed mappers, pgvector integration via `pgvector-python` |
| Migrations | Alembic, `async` env, data migration for seeding | Standard + schema + data in one tool |
| Embeddings | `fastembed` (ONNX int8) w/ `intfloat/multilingual-e5-small` | Multilingual covering Spanish, 120 MB int8, ~30–60 ms/query on Haswell |
| Vector search | pgvector w/ `ivfflat (vector_cosine_ops) WITH (lists=100)` | Postgres stays single source of truth; flat-ish perf at current scale, scales fine to 100k |
| LLM runtime | llama.cpp `llama-server` (Docker `ghcr.io/ggml-org/llama.cpp:server`) | Fastest CPU path, native JSON-schema grammar, OpenAI-compat endpoint |
| LLM model | `Qwen3-4B-Instruct-2507` Q5_K_M (+ `Qwen3-4B-Thinking-2507` at tier 4) | Apache-2.0, strong Spanish + BFCL; see context.md §4.3 |
| LLM client | `openai` (`AsyncOpenAI` pinned to local base_url) | Single well-tested lib; swappable later |
| Tools format | Hermes (native Qwen3) via `tool_choice: "auto"` | Qwen-recommended |
| Structured output | JSON-Schema via llama.cpp grammar (sampler-level) | Model cannot emit invalid slug; no post-hoc validation needed |
| Observability | Prometheus scrape + structlog JSON logs | Matches existing VPS stack; bounded-cardinality counters/histograms |
| Auth | Traefik `apitokenmiddleware` plugin, bearer `vault_categorizer_api_token` | Plugin already installed repo-wide; pattern reused from `obs-bearer` |
| Deploy | Jenkins builds → Zot registry → Ansible-managed Compose on VPS | Same flow as miplata; see §3 |
| Container runtime | Docker Compose v2 (`community.docker.docker_compose_v2`) | Repo-wide default |

## 3. Deploy topology

### 3.1 Networks

| Network | External? | Members (as of now) |
|---|---|---|
| `proxy` | yes | traefik, miplata-web, metabase, grafana proxy, pdns-admin, categorizer-api |
| `miplata_db` | yes | miplata-postgres, categorizer-api (for read-only access) |
| `observability` | yes | prometheus, loki, grafana, cadvisor, postgres-exporter, categorizer-api (for metrics scrape), future llama-server metrics |
| `categorizer_net` | yes (new) | llama-server, categorizer-postgres, categorizer-api |

### 3.2 Ansible roles

| Role | Host | Key responsibilities |
|---|---|---|
| `llm_inference` | main vps | Creates `/opt/llm_inference/`, renders env + compose, starts llama-server. Does NOT pull GGUF (operator-placed). |
| `categorizer` | main vps | Creates `/opt/categorizer/`, renders env + compose, adds DNS A record, tolerant pull from Zot, enables pgvector extension post-up. |
| `miplata` (modified) | main vps | Now renders `CATEGORIZER_BASE_URL` and `CATEGORIZER_API_TOKEN` into miplata's `.env`. |
| `jenkins` (modified) | jenkins vps | JCasC now has `github-categorizer-ssh-key` credential + `categorizer-deploy` pipelineJob. |

### 3.3 Image/tag ownership

| Image | Registry | Source | Pull policy |
|---|---|---|---|
| `ghcr.io/ggml-org/llama.cpp:server` | GitHub Container Registry (public) | upstream | `pull: missing` (ansible) |
| `pgvector/pgvector:pg16` | Docker Hub (public) | upstream | `pull: missing` |
| `registry.web.<domain>/miplata/categorizer:latest` | self-hosted Zot | built by `categorizer-deploy` Jenkins job | `pull: always` (compose) |

### 3.4 The Jenkins flow

1. User (or the `git push` webhook) triggers the `categorizer-deploy` Jenkins job.
2. Stages: `Lint & Typecheck` (ruff) → `Test` (pytest) → `Build & Push Image` (docker build + push to Zot) → `Deploy to VPS` (SSH + `docker compose pull && docker compose up -d`) → `DB Migrate` (SSH + `docker compose run --rm api alembic upgrade head`).
3. DB Migrate runs both `0001_initial` and `0002_seed_own_accounts_and_merchants`. Post-migration the service is live with 6 own_accounts and 42 merchants pre-populated.
4. The retrieval index is still empty at this point — population happens in §4.5.

## 4. Cascade tier detail

### 4.1 Tier 1 — normalize

- **File:** `src/categorizer/normalize.py`
- **Status:** ✅ complete; 40+ patterns, 40 tests pass against real Bancolombia/Nequi strings.
- **Key invariants:**
  - Pure function, no randomness, no external state.
  - Emits **sentinel tokens** with underscores (e.g. `gmf_4x1000`, `envio_persona`, `retiro_cajero`) that tier 3 matches against. Underscores MUST survive the final punctuation cleanup — fix normalize.py, not rules.py, if rules stop matching.
  - Preserves `ñ` (distinguishes `baño` from `bano`).
  - Strip ordering is most-specific-first (e.g. `CXC IMPT GOBIERNO 4X1000 MON` before `IMPT GOBIERNO 4X1000`).
- **Known gap:** dots are stripped, so rules in `rules.py` must not rely on them (`claude.ai` → `claude ai`).

### 4.2 Tier 1.5 — deterministic rules

- **File:** `src/categorizer/rules.py`
- **Status:** ✅ complete; 26 tests pass.
- **Two paths** (both hit at confidence 0.95+):
  1. **Internal-transfer by text** — `resolve_internal_transfer_by_text` matches the sentinel tokens emitted by normalize against the `own_accounts.aliases` set.
  2. **Fixed merchant/finance rules** — an ordered list of `(pattern, slug, reasoning)` tuples in `_MERCHANT_RULES`. Order matters: more specific first (e.g. `didi food` before bare `didi`).
- **Contract:** only returns a `RuleHit` at `confidence ≥ 0.95`; otherwise returns `None` and the cascade escalates. Keep the tier binary — don't add 0.80-confidence rules.

### 4.3 Tier 2 — features (not a classifier, but pipeline input)

- **File:** `src/categorizer/features.py`
- **Status:** ✅ complete (pure helpers).
- Produces temporal, amount-bucket, and recurring-match signals that tier 3 sees as context (not yet fed into the LLM prompt — follow-up).

### 4.4 Tier 2.5 — paired-tx internal transfer

- **File:** `src/categorizer/cascade.py` → `_paired_internal_transfer`
- **Status:** ⚠️ partial. Implementation is there, but the matching predicate compares `account_slug` between the incoming tx and labeled tx. miplata passes its Prisma UUID as `account_slug`, while our `own_accounts` uses friendly slugs (see context.md §5.10). Until the UUID sync follow-up lands, this tier won't find pairs. Rules tier still handles the text-based variants, which is >80% of real cases.

### 4.5 Tier 3 — merchant lookup

- **File:** `src/categorizer/merchant.py`
- **Status:** ✅ v1 complete (exact + token-contains match over the seeded dict).
- **Carcass:** fuzzy match (RapidFuzz / trigram) for novel merchants — marked TODO in the file. Falling through to retrieval + LLM is acceptable until the volume of unresolved-merchant rows warrants the build.

### 4.6 Tier 4 — pgvector kNN retrieval

- **File:** `src/categorizer/retrieval.py`
- **Status:** ✅ complete; requires a non-empty `labeled_transactions.embedding` column to do anything useful.
- **Seeding path:** `scripts/seed_gold_set.py` hits `/v1/label` for each row in `config/gold_set_v1.jsonl`. Each label triggers `fastembed.embed` (first call downloads the model into `/app/artifacts/embedding_cache/`) and writes the row.
- **Voting:** distance-weighted vote across top-k=8 neighbors; thresholds: top-1 similarity ≥ 0.85 AND margin (top-1 minus top-1-of-different-class) ≥ 0.05.

### 4.7 Tier 5 / 6 — LLM (no-think + think)

- **File:** `src/categorizer/llm.py` (client) + `src/categorizer/tools.py` (tool schemas + dispatch) + `src/categorizer/cascade.py` (orchestration).
- **Status:** ✅ complete end-to-end, but requires llama-server on `categorizer_net` — already deployed.
- **Request shape:** `temperature=0`, `top_p=1`, `seed=42`, `response_format: json_schema` with an enum of `taxonomy.implemented_slugs` (NOT all slugs — carcasses are excluded from allowed outputs to prevent false positives on unimplemented categories). Tools passed as `tools=[…]` with `tool_choice="auto"` on the /no_think pass; no tools on the follow-up and on /think pass.
- **Tool loop:** single round only. If the first call returns tool_calls, we dispatch them, append their results as `role: "tool"` messages, and do one final call. This matches research guidance — Qwen3-4B at Haswell AVX2 quality degrades past one tool round.

### 4.8 Tier 7 — reject

- **Behavior:** when no tier hit its threshold, return `sin_clasificar.pendiente` with `source="llm_notink"` and the last confidence value. Audit row is still written to `predictions`.
- **The correct state**, not a degraded one. Rejected rows surface via `GET /v1/uncertain` for the user to review. Every user correction via `POST /v1/label` feeds the retrieval index and (later) the fine-tuning corpus.

## 5. Status matrix

### 5.1 Repository-level

| Item | Status | Notes |
|---|---|---|
| `vps-infrastructure` roles (`llm_inference`, `categorizer`) | ✅ deployed | Run from the worktree; commit is on `worktree-ai-model` branch (not yet pushed/merged). |
| `vps-infrastructure` JCasC + env updates | ✅ deployed | Jenkins restarted; new categorizer job + credential visible. |
| `miplata` categorization refactor | ✅ committed local on `feat/observability-instrumentation` | Commit `f10a536`. NOT pushed (see context.md §5.1). |
| `miplata-categorizer` scaffold + seed + Jenkinsfile + docs | ✅ pushed to `origin/main` | 4+ commits. |
| Vault entries | ✅ added and encrypted | `vault_categorizer_db_password`, `vault_categorizer_api_token`, `vault_jenkins_categorizer_ssh_key`. |
| GGUF on VPS | ✅ at `/opt/llm_inference/models/Qwen3-4B-Instruct-2507-Q5_K_M.gguf`, 2.7 GB | Owned by `ubuntu:ubuntu`. |
| GitHub deploy key for Jenkins | ✅ added to `GonzMenSeb/miplata-categorizer` | Read-only; private half in vault. |
| GitHub push webhook | ✅ installed on `miplata-categorizer` | Points at `jenkins.web.<domain>/github-webhook/`. |
| First Jenkins `categorizer-deploy` build | 🟡 in flight (empty commit pushed) | Expected outcome: image in Zot, categorizer stack up, migrations applied. |

### 5.2 Component-level (`src/categorizer/`)

| Module | Status | Notes |
|---|---|---|
| `taxonomy.py` | ✅ | Loads YAML, exposes enum for schema generation + validity checks. |
| `normalize.py` | ✅ | 40+ regex patterns, Colombian-specific. Ñ preserved. |
| `rules.py` | ✅ | Covers ~20 determinable patterns + internal-transfer-by-text. |
| `features.py` | ✅ | Temporal + amount buckets + recurring detection (pure helper). |
| `merchant.py` | 🟡 partial | Exact + substring match on seeded dict (15 entries in file + 42 seeded via migration). Fuzzy-match carcass documented. |
| `retrieval.py` | ✅ | pgvector kNN + distance-weighted voting. Requires seeded embeddings to be useful. |
| `llm.py` | ✅ | `AsyncOpenAI` against llama-server, JSON-schema grammar, `/think` toggle. |
| `tools.py` | ✅ | 4 tools, real DB-backed dispatch for all 4. |
| `cascade.py` | ✅ | 9-tier orchestrator. Paired-tx tier works against labeled tx only (see §4.4 gap). |
| `storage.py` | ✅ | 5 tables + IVFFlat index + async engine + session factory. |
| `api.py` | ✅ | `/v1/categorize`, `/v1/label`, `/v1/uncertain`, `/v1/own-accounts`, `/healthz`, `/metrics`. |
| `metrics.py` | ✅ | 5 Prometheus series, bounded cardinality. |
| `config.py` / `logging_setup.py` / `main.py` | ✅ | pydantic-settings, structlog JSON, FastAPI lifespan loads taxonomy. |
| `alembic/versions/0001_initial.py` | ✅ | Full schema + `CREATE EXTENSION vector`. |
| `alembic/versions/0002_seed_own_accounts_and_merchants.py` | ✅ | 6 own_accounts + 42 merchants, idempotent upsert. |
| `config/taxonomy.yaml` | ✅ | 13 parents × 54 children, `implemented: false` on 21 carcass children. |
| `config/gold_set_v1.jsonl` | ✅ | 50 hand-labeled transactions from real Bancolombia + Nequi strings. |
| `scripts/seed_gold_set.py` | ✅ | Idempotent POST-each-row to `/v1/label`. Not yet run against prod. |
| `tests/test_normalize.py` + `tests/test_rules.py` | ✅ 70 pass | No async/DB tests yet (need pytest-asyncio + testcontainers — follow-up). |
| `Dockerfile` | ✅ | Multistage python:3.12-slim, non-root runtime. |
| `infra/jenkins/Jenkinsfile` | ✅ | 5 stages, requires `docker-workflow` plugin (installed). |

### 5.3 Not yet implemented (carcass / follow-ups)

| Item | Where | Priority |
|---|---|---|
| Fine-tune pipeline (Unsloth QLoRA) | new `training/` dir | ⚠️ scheduled for when labeled corpus ≥ 500 |
| BERT-ES fast-path classifier | new `src/categorizer/fast_path.py` | nice-to-have; retrieval + rules hit >70% already |
| Merchant fuzzy match (RapidFuzz) | `merchant.py` | marked TODO |
| Paired-tx UUID sync / alias enrichment | see §7.2 | blocks paired-tx internal-transfer tier |
| `MIPLATA_RO_DATABASE_URL` actually-used read path | cascade.py `_paired_internal_transfer` TODO | enables pair detection against unlabeled miplata tx |
| Prometheus scrape config for categorizer + llama-server | `vps-infrastructure/roles/monitoring/` | add targets, dashboard follows |
| Grafana "Categorizer Health" dashboard | `vps-infrastructure/roles/monitoring/templates/dashboards/` | tier hit rate, confidence histogram, per-parent F1 |
| Alert rules | same role | accuracy drop, ECE drift, reject-rate spike |
| Gold-set eval script | `eval/eval.py` (new) | compute macro-F1, per-parent F1, hierarchical F1, adaptive ECE |
| Shadow-mode A/B harness | `eval/shadow.py` | optional, only if we reintroduce an API-based ground-truth tier |
| Active-learning UI in miplata | `miplata/apps/web/src/…` | surfaces `/v1/uncertain` + correction flow |
| `Category` table retirement in miplata | Prisma migration | coordinate with UI changes |
| English-taxonomy removal | miplata code | last in order |

## 6. Configuration surface

### 6.1 Environment variables consumed by the categorizer

| Name | Source | Default | Purpose |
|---|---|---|---|
| `CATEGORIZER_ENV` | env | `production` (via ansible) | Labels logs / metrics. |
| `CATEGORIZER_PORT` | env | `8000` | FastAPI port. |
| `CATEGORIZER_LOG_LEVEL` | env | `INFO` | structlog filter level. |
| `DATABASE_URL` | env | — | Async psycopg DSN into categorizer's own Postgres. |
| `MIPLATA_RO_DATABASE_URL` | env | — | Read-only DSN into miplata's Postgres (follow-up use). |
| `LLM_BASE_URL` | env | `http://llama-server:8080/v1` | llama-server OpenAI-compat endpoint. |
| `LLM_MODEL` | env | `qwen3-4b-instruct-2507` | Passed as `model` in `chat/completions` requests. |
| `CATEGORIZER_API_TOKEN` | env (Traefik middleware) | — | Bearer token clients must present. |
| `CATEGORIZER_RULE_MIN_CONFIDENCE` | env | `0.95` | Rule tier floor. |
| `CATEGORIZER_KNN_MIN_CONFIDENCE` | env | `0.85` | kNN top-1 floor. |
| `CATEGORIZER_KNN_MIN_MARGIN` | env | `0.05` | kNN margin floor. |
| `CATEGORIZER_LLM_MIN_CONFIDENCE` | env | `0.70` | LLM output floor. |
| `CATEGORIZER_THINK_TRIGGER_CONFIDENCE` | env | `0.60` | Below this, escalate to /think. |
| `CATEGORIZER_TAXONOMY_PATH` | env | `/app/config/taxonomy.yaml` | Where the YAML lives in the container. |
| `CATEGORIZER_ARTIFACTS_DIR` | env | `/app/artifacts` | Embedding cache + any future persisted state. |
| `CATEGORIZER_EMBEDDING_MODEL` | env | `intfloat/multilingual-e5-small` | fastembed model name. |
| `CATEGORIZER_EMBEDDING_DIM` | env | `384` | Must match model; vector column is `Vector(dim)`. |

### 6.2 llama-server env (role `llm_inference`)

All `LLAMA_ARG_*` variables, see `roles/llm_inference/templates/env.j2`. Key knobs documented in context.md §5.14.

## 7. Remaining work — prioritized

### 7.1 (c) Fine-tuning pipeline — when corpus ≥ 500 labeled tx

**Goal:** QLoRA-tune Qwen3-4B-Instruct-2507 on Sebas's corpus + a slice of general instruction data; merge → Q4_K_M GGUF; swap into llama-server.

**Concrete plan:**
1. New dir `training/` in this repo.
2. `training/dataset.py` — pull labeled rows from `labeled_transactions` (or a pg_dump snapshot), format as Qwen3 ChatML JSONL with the exact system+user prompt used at inference time, hold out 10% as eval.
3. `training/train.py` — Unsloth + PEFT recipe: `r=16`, `lora_alpha=32`, `target_modules="all-linear"`, `lr=2e-4`, cosine, 3 epochs, `bf16=True` (not QLoRA 4-bit — Qwen3.5 had quant issues; Qwen3 is fine but bf16 LoRA is cleaner at 24 GB), mix 10% Tulu-3 for instruction-preservation.
4. Rent RunPod Community Cloud RTX 4090 (~$0.34/hr, 3-4 hr per run, **~$1.50 total**).
5. Merge LoRA → bf16 → GGUF via `llama.cpp/convert_hf_to_gguf.py` → `llama-quantize Q4_K_M`.
6. Copy the merged GGUF to `/opt/llm_inference/models/qwen3-4b-miplata-v1.gguf`, update `llm_inference_model_file` in group_vars, re-run Ansible.
7. A/B test: run both v0 (base) and v1 (tuned) on the gold set; promote if macro-F1 improves ≥ 2 pp **and** MMLU-100 canary doesn't drop > 3 pp.
8. Cadence: re-train weekly while the label stream is heavy, monthly thereafter.

**Hard constraint:** re-baseline against fresh Claude Sonnet on the gold set every 3 months to ensure we're still beating it on Sebas's domain. If we're not, the architecture assumption is wrong and we revisit.

### 7.2 Paired-tx internal-transfer fix

**Goal:** enable tier 4 (paired-tx detection) for real traffic.

**Concrete plan (pick one):**
- **Option A (cleanest):** add a `miplata_account_id UUID` column to `own_accounts`, populate via a one-shot script that queries `MIPLATA_RO_DATABASE_URL` and updates each of the 6 rows. Change `_paired_internal_transfer` to match on `account_slug = miplata_account_id`.
- **Option B:** update `_paired_internal_transfer` to also accept matches on `aliases @> ARRAY[:account_slug]`, and stuff each miplata account UUID into the `aliases` array of the corresponding row via the same one-shot script.
- **Option C:** introduce a translation endpoint `POST /v1/own-accounts/sync` that miplata calls with its Account list on app startup, populating own_accounts from miplata's truth.

My recommendation: **Option A** — least special-casing, scales if a new app starts using the categorizer.

### 7.3 (d) Eval harness

**Goal:** measure hierarchical macro-F1, adaptive ECE per tier, tier hit rates, and mean correction distance, against a growing gold set.

**Concrete plan:**
1. New dir `eval/`.
2. `eval/eval.py`: reads `config/gold_set_v1.jsonl`, runs each through `/v1/categorize` (or an in-process cascade for speed), computes: flat accuracy, macro-F1, per-parent F1, `hiclass` hierarchical F1, adaptive ECE (via `netcal`) per tier, tier hit-rate histogram. Writes `metrics.json` + a Prometheus textfile.
3. `eval/compare.py`: McNemar exact test + paired bootstrap vs the previous version's metrics — used as a promotion gate in the Jenkinsfile.
4. Grow the gold set: every 2–4 weeks, sample 20–50 recent-corrected rows and add to `gold_set_v2.jsonl`, etc. Freeze v1 forever as a regression baseline.

### 7.4 (e) Carcass fill-ins

In priority order (highest expected impact first):

1. **`hogar.arriendo`** — detect large (>$1M COP) monthly recurring debit to the same recipient. Straightforward: use `features.detect_recurring` + amount-bucket filter. Rule in `rules.py`.
2. **`salud.farmacia`** — add patterns for `CRUZ VERDE`, `FARMATODO`, `DROGAS LA REBAJA`, `LOCATEL`, etc. Update merchants table + rules.
3. **`hogar.internet_telefonia` amount-based disambiguation** — currently bucketed with telcos; for "COMPRA PSE EN CLARO" we could disambiguate mobile-recharge vs postpaid-bill by amount.
4. **`ocio.bares`** — harder; typically requires the merchant dict. Seed a few common Medellín bars over time.
5. **`compras.ropa` / `compras.electronica`** — needs merchant dict growth; not rule-resolvable cleanly.

### 7.5 Active-learning UI in miplata

**Goal:** surface `GET /v1/uncertain` in the miplata UI with a single-tap correction affordance, plus a weekly-digest view.

Out of scope for this repo — tracked as a miplata follow-up.

### 7.6 Monitoring + dashboards

1. Add scrape targets for `categorizer-api:8000/metrics` and `llama-server:8080/metrics` in `vps-infrastructure/roles/monitoring/templates/prometheus.yml.j2`.
2. Provision a `categorizer-health.json` Grafana dashboard with: tier hit rate (stacked area), macro-F1 (24h), adaptive ECE per tier, reject rate, Claude-free (always 0 — it's the policy).
3. Alert rules: `CategorizerAccuracyDrop`, `CategorizerCalibrationDrift`, `CategorizerRejectSpike`.

## 8. Open questions / decisions for Sebas

| Question | Default if not answered | Where it bites |
|---|---|---|
| How do we sync miplata account UUIDs into the categorizer's own_accounts? | Option A in §7.2 | Blocks paired-tx tier. |
| When to push the unpushed commits on `miplata/feat/observability-instrumentation` and `vps-infrastructure/worktree-ai-model`? | User's call — not blocking local work. | Triggers Jenkins builds; triggers fresh deploys. |
| Do we start the fine-tune pipeline before or after the BERT-ES fast path? | Fine-tune first (bigger accuracy lift). | Research agents suggested BERT-ES first; real-world tradeoff depends on corpus growth rate. |
| Do we re-baseline against Claude Sonnet periodically? | Yes, every 3 months, per §7.1. | Sanity check the "beats Claude" hypothesis. |
| Who holds the fine-tune VPS rental budget? | Sebas, ~$10–20 / month. | Decide before first training run. |

## 9. How to verify current state (repeatable checks)

### 9.1 From anywhere

```bash
# llama-server health
ANSIBLE_VAULT_PASSWORD_FILE=/home/sebastian/versioned-code/vps-infrastructure/.vault_pass \
  ansible vps -e "@/home/sebastian/versioned-code/vps-infrastructure/vault.yml" \
  -i /home/sebastian/versioned-code/vps-infrastructure/inventory.yml \
  -m shell -a "docker exec llama-server curl -fsS http://127.0.0.1:8080/health"

# categorizer health (once Jenkins build lands)
TOKEN=$(ANSIBLE_VAULT_PASSWORD_FILE=/home/sebastian/versioned-code/vps-infrastructure/.vault_pass \
  ansible-vault view /home/sebastian/versioned-code/vps-infrastructure/vault.yml \
  | grep vault_categorizer_api_token | awk -F'"' '{print $2}')
curl -fsS -H "Authorization: Bearer $TOKEN" https://categorizer.web.vespiridion.org/healthz

# end-to-end categorization
curl -fsS -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -X POST https://categorizer.web.vespiridion.org/v1/categorize \
  -d '{"transaction":{"external_id":"probe-1","account_slug":"nequi","date":"2025-12-31","amount":-38900,"currency":"COP","description":"PAGO EN QR BRE-B: Toretos","transaction_type":"debit"},"return_trace":true}'
```

### 9.2 From the repo

```bash
cd /home/sebastian/versioned-code/miplata-categorizer
python3.12 -m venv .venv && .venv/bin/pip install '.[dev]'
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v    # expect 70 pass
```

## 10. Common maintenance operations

### 10.1 Rotating the API token

1. Generate new token: `openssl rand -hex 32`.
2. `ansible-vault decrypt vault.yml` (with `ANSIBLE_VAULT_PASSWORD_FILE=.vault_pass`).
3. Replace `vault_categorizer_api_token`.
4. `ansible-vault encrypt vault.yml`.
5. Re-run `ansible-playbook playbook.yml` → re-renders the compose label → `docker compose up -d` recreates the container with the new token.
6. Update miplata's env (via another playbook run if the token lives there too — it does).

### 10.2 Bumping the LLM model

1. Download new GGUF to `/opt/llm_inference/models/<name>.gguf`.
2. Update `llm_inference_model_file` in `roles/llm_inference/defaults/main.yml`.
3. `ansible-playbook playbook.yml` → re-renders env → `docker compose up -d` recreates llama-server with new model.
4. If context size or parallel slots change, tune `LLAMA_ARG_CTX_SIZE` / `LLAMA_ARG_N_PARALLEL` proportionally.

### 10.3 Adding a new category

1. Add the entry in `config/taxonomy.yaml`. Set `implemented: true` if you have at least 1 rule or ~20 seeded labels for it; otherwise `implemented: false` (carcass).
2. If it's rule-addressable, add a pattern in `rules.py`'s `_MERCHANT_RULES`. Add test cases in `tests/test_rules.py`.
3. Commit, push → Jenkins builds → categorizer restarts. **Alembic migration NOT needed** (taxonomy is YAML, not DB).

### 10.4 Adding an own_account (e.g. Sebas opens a new credit card)

Two options:
- **Ansible path:** add a new tuple to `_OWN_ACCOUNTS` in a new Alembic migration `0003_…py`, re-deploy.
- **Live path:** `POST /v1/own-accounts` (endpoint exists only for GET today — adding POST is a small follow-up).

### 10.5 Retraining / refreshing the merchant dict

- **Ansible path:** append to `_MERCHANTS` in a new Alembic migration.
- **Learned path (future):** every N user corrections where the user maps the same raw-merchant-substring to the same category, auto-insert a `source="learned"` row into `merchants`.
