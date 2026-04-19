# context.md — full context for the miplata-categorizer project

> Purpose: catch a future reader (human or Claude Code) up on *everything* about
> this project — why it exists, how we got here, and all the non-obvious bits
> that are easy to lose. For the "how to build it / what's left" view, read
> `plan.md`. For the cascade internals, read the module docstrings in `src/`.

---

## 1. One-paragraph summary

**miplata-categorizer** is a self-hosted HTTP service that takes a Colombian-Spanish bank-statement transaction line and returns a category slug, a confidence score, and (on request) the internal reasoning trace. It runs entirely on Sebas's existing OVH VPS with no third-party API in the hot path. It replaces a "WAY primitive" (Sebas's words) Claude-SDK wrapper that was previously wired into miplata's API but never actually enabled in production. The goal is not to match a stateless Claude-Sonnet call — it's to *beat* one on this specific domain by doing four things a stateless call cannot: retrieve from the user's own labeled history, consult a Colombian-merchant dictionary, detect movements between the user's own accounts via deterministic rules, and continuously learn from user corrections.

## 2. The problem we're solving

1. **miplata** is Sebas's personal-finance app (TypeScript/NestJS + Vite/React, Prisma + Postgres + Redis, deployed to the same OVH VPS via Jenkins + a self-hosted Zot registry). Users import Colombian bank statements (Bancolombia savings + credit cards, Nequi) and the app extracts transactions. Every transaction needs a category.

2. The categorization path as-shipped was: a shallow rule-matcher, then a Claude-Sonnet call with a 30-transaction batch prompt and a small English-only taxonomy. The prompt passed only `{description, amount}`. The response was a loose JSON array parsed by regex. There was no prompt caching, no tool use, no few-shot from the user's own label history, no confidence calibration, silent failure to an empty array on error. In production, `ANTHROPIC_API_KEY` was never even rendered into miplata's `.env`, so the Claude branch was *never actually executed*. Every unmatched transaction fell through to uncategorized.

3. Sebas wants something genuinely better. Not a drop-in replacement for the primitive code. A system that (a) costs effectively nothing to run, (b) keeps transaction data on his own hardware, (c) handles Colombian-Spanish nuance (Nequi BRE-B prefixes, GMF 4x1000, anonymized counterparty names, PSE payments, Cívica transit reloads), and (d) explicitly distinguishes transfers between the user's own accounts from spending and from P2P transfers — a first-class concept the old system lacked.

## 3. Why self-host (and the tradeoffs we accepted)

We considered five alternatives in detail (see the research agents' outputs from the planning phase). The per-month cost at 3k–10k tx/month comes out:

| Option | 10k tx/month | Latency/tx | Ops burden | Data leaves? |
|---|---|---|---|---|
| Claude Haiku 4.5 + prompt cache | ~$3.40 | ~1 s | low | yes |
| OpenAI GPT-5 nano | ~$0.41 | ~1 s | low | yes |
| DeepInfra Llama 3.1 8B Turbo | ~$0.11 | ~1 s | low | yes (3rd party) |
| Self-host on existing VPS (chosen) | $0 | 5–15 s | medium | **no** |
| Dedicated-GPU VPS (Hetzner GEX44) | ~$205 | <0.5 s | med-high | no |

At Sebas's volume, self-host does not win on pure dollar ROI — Haiku with caching is $3.40/mo and payback on ~40 engineering hours is measured in years. **Self-host wins when the objective function includes:**

1. **Data privacy / self-sovereignty** — Sebas's explicit stance is "self-hosted only, I want to see how far this goes." No API fallback in production, even for the hard tail.
2. **The learning value of building it** — explicit user intent.
3. **A system that can beat stateless API calls by specializing** — the categorizer has access to the user's own labels, a Colombian merchant dictionary, and internal-transfer detection rules that a Claude API call cannot see.
4. **Predictable fixed cost** — $0 marginal cost per transaction; the VPS is already paid.

We deliberately accepted: slower per-tx latency (5–15 s on Haswell CPU vs ~1 s via API), a batch-friendly design instead of real-time, an ~80% primary-category accuracy target in the first month (ramping with corrections), and a nontrivial engineering budget of ~40 hours for v1.

## 4. How we arrived at the current architecture

### 4.1 Hardware reality check (the binding constraint)

Sebas's VPS is an **OVH Value-tier KVM instance**:
- CPU: `Intel Core Processor (Haswell, no TSX)`, 6 vCPU QEMU-virtualized exposed as 6 sockets × 1 core × 1 thread at 2.0 GHz.
- Flags: `avx avx2 fma f16c aes bmi1 bmi2 sse4_2` — **no AVX-512, no VNNI, no BF16, no AMX.**
- RAM: 11 GB DDR4 (dual-channel, speed unadvertised — probably 2400/2666). 8.3 GB available at baseline with existing services running.
- Swap: 2 GB (thin — OOM risk if observability stack spikes + LLM loads).
- Disk: 96 GB total, 67 GB free.

Haswell is 2013-era silicon. Memory bandwidth, not FLOPs, is the bottleneck for LLM inference. Realistic throughput: **~6 tok/s generation** on a 4B Q4_K_M model, ~25 tok/s prompt eval. So a ~40-token JSON response after a cached ~500-token system prompt lands at ~6–10 s/tx. That's too slow to put an LLM in the critical path of every transaction. The architecture must therefore funnel most traffic away from the LLM.

### 4.2 The cascade insight

Research converges on the same answer for this problem shape: a 4-tier cascade. Each tier handles one band of difficulty and emits a calibrated confidence score; only low-confidence outputs escalate.

| Tier | Target share | Tech | Typical latency |
|---|---|---|---|
| 1. Deterministic rules | ~40% | regex over normalized text + paired-tx check against own_accounts | 5 ms |
| 2. kNN retrieval | ~30% | pgvector + multilingual-e5-small embedding | 50 ms |
| 3. Small LLM with tools | ~25% | Qwen3-4B-Instruct-2507 Q5_K_M via llama.cpp + Hermes tool parsing | 7–10 s |
| 4. Small LLM in /think mode | ~5% | Qwen3-4B-Thinking-2507 re-pass when tier 3 confidence is low | 15–25 s |

Anything below threshold after tier 4 is rejected to `sin_clasificar.pendiente` (no API fallback per Sebas's policy). The user reviews those manually; every correction feeds back into tiers 1, 2, and (later) the fine-tuning data for a future tier-3 upgrade.

### 4.3 Why these specific stack picks

- **Qwen3-4B-Instruct-2507** as the base model. Apache-2.0 (no Llama/Gemma commercial caveats), multilingual training on 36T tokens over 119 languages (Spanish is a first-class language for it), strong BFCL-v3 score (61.9 instruct / 71.2 thinking) — the best sub-8B model published as of early 2026 for tool use.
- **Q5_K_M quantization**, not Q4_K_M. RAM fits comfortably (~3 GB weights + 0.5 GB KV at 2K ctx), perplexity regression negligible, and Haswell AVX2 doesn't gain meaningfully from going smaller.
- **llama.cpp's `llama-server`** as the runtime, not Ollama. ~20% faster than Ollama on CPU for single-user workloads, native JSON-schema grammar (the model *cannot* emit an invalid category slug), and KV-cache reuse across the same system prompt.
- **pgvector in a dedicated Postgres** (not shared with miplata's). The categorizer owns its data end-to-end — own_accounts, labels, merchants, predictions, corrections. Reused embeddings live in the same row as the labels that produced them. Backups are a single `pg_dump`.
- **intfloat/multilingual-e5-small via fastembed (ONNX int8)** for embeddings. 118M params, 384-dim, ~120 MB int8, proven Spanish retrieval quality, 30–60 ms per query on Haswell.
- **Python 3.12 + FastAPI + SQLAlchemy 2.0 async** for the service. Standard Python stack, plays nicely with pgvector, Pydantic v2 gives us strict request/response shapes, Alembic handles migrations including data seeding.

### 4.4 The first-class concept we added: *movimientos entre cuentas propias*

Sebas explicitly asked that transfers between his own accounts be categorized differently from spending and from P2P transfers. This is the single biggest contract difference from the old miplata taxonomy. It's a detection problem that rules — not an LLM — handle best, because the signal is structural (institution name matches an own_account, or a paired transaction of opposite sign + same amount appears on another own_account within ±2 days). The categorizer's `own_accounts` table is the ground truth; rules check against it deterministically.

### 4.5 Why not fine-tune now

Fine-tuning Qwen3-4B on Sebas's labels is in scope but deliberately deferred past v1. We need ~500+ labeled transactions before fine-tuning measurably beats zero-shot, and Sebas's current labeled corpus is ~50 (the hand-labeled gold set we seeded). The right sequence is: ship the cascade, accumulate labels for 4-8 weeks via real usage + corrections, then fine-tune on ~2000+ rows. The plan document tracks this as a roadmap item.

## 5. Non-obvious things worth knowing

### 5.1 Repo layout — three coupled repos, branch state as of 2026-04-19

| Repo | Path on Sebas's laptop | Purpose | Remote |
|---|---|---|---|
| `vps-infrastructure` | `/home/sebastian/versioned-code/vps-infrastructure/` | Ansible IaC for both VPS hosts | `git@github.com:GonzMenSeb/vps-infrastructure.git` |
| `miplata` | `/home/sebastian/versioned-code/miplata/` | The personal-finance app (NestJS + React) | `git@github.com:GonzMenSeb/miplata.git` |
| `miplata-categorizer` | `/home/sebastian/versioned-code/miplata-categorizer/` | This project (FastAPI cascade) | `git@github.com:GonzMenSeb/miplata-categorizer.git` |

**Exact branch state at end of this build-out session (critical for a fresh session to pick up from):**

| Repo | Active working tree | Commit(s) I added | Push state | What to do next |
|---|---|---|---|---|
| `miplata-categorizer` | `main` (clean) | 4 commits (initial scaffold, seed + Jenkinsfile, ci-trigger, docs) | **pushed to `origin/main`** — no outstanding work | — |
| `vps-infrastructure` | worktree at `.claude/worktrees/ai-model/` on branch `worktree-ai-model` | 1 commit: `feat(infra): add llm_inference + categorizer roles…` (sha starts `e9bf12…`) | **local only on `worktree-ai-model`**. `main` is at a different sha (`ed9a0ee…`) ahead of the worktree's fork point on unrelated docs commits; merging is a separate decision. **Not pushed.** | Pick: (a) merge `worktree-ai-model` → `main` locally with a merge commit and push main, or (b) push the `worktree-ai-model` branch as-is and PR to main. Do NOT force-push main. |
| `miplata` | branch `feat/observability-instrumentation` | 1 commit: `feat(categorization): replace primitive Claude classifier…` (sha starts `f10a536…`) | **local only on `feat/observability-instrumentation`**. `master` is 7 commits *behind* `origin/master`; local `master` is stale. **Not pushed.** | Pick: (a) push `feat/observability-instrumentation` to its origin (already tracks), merge/rebase onto master in a PR; or (b) checkout `master`, `git pull origin master`, cherry-pick `f10a536`, push master. Either way, **do not try to merge master locally without syncing origin first.** |

Both unpushed commits are meaningful and reviewed — they're held back only because a cohesive push would require resolving unrelated local branch state, which is outside this session's scope. **The `llm_inference` + `categorizer` roles are already deployed on the VPS** (Ansible was run from the worktree), so the on-disk state of the infra is ahead of both remotes. A fresh session doesn't need to re-run Ansible to pick up — it only needs to push, or adopt the commits, if it wants them reflected on the remote.

### 5.2 The two VPS hosts

Two separate OVH Value-tier boxes defined in `inventory.yml`:

- **`vps`** — the public web host. Runs Traefik + Let's Encrypt, PowerDNS (authoritative for the project domain), monitoring stack (Prometheus, Loki, Grafana, Alloy, cAdvisor, blackbox-exporter), Vaultwarden, Zot registry, miplata (API + web + Postgres + Redis), Metabase, and now `llm_inference` + `categorizer`. SSH user: `ubuntu`. Public IP in vault as `vault_main_vps_ip`.
- **`jenkins`** — CI/CD host. Runs only `jenkins-ansible:latest` (locally-built from `roles/jenkins/files/Dockerfile`) + JCasC config + observability agent. SSH user: `root` (different from main VPS — templates that need a username should read `ssh_allowed_users`). Public IP in vault as `vault_jenkins_vps_ip`.

Play order in `playbook.yml` matters: `traefik` must run before any role joining the `proxy` network. `dns` runs before `miplata`/`categorizer` because those roles call `pdnsutil` in the pdns container.

### 5.3 Docker network topology

Four external Docker networks exist, created once by `base`/`traefik`:
- `proxy` — Traefik's routing plane. Every user-facing service joins this.
- `miplata_db` — isolates miplata's Postgres + Redis from `proxy`. The categorizer also joins this to read miplata.transactions as `miplata_ro_user`.
- `categorizer_net` (new) — isolates the categorizer's Postgres + the llama-server from everything else. categorizer-api + llama-server + categorizer-postgres are all here.
- `observability` — scraped by Prometheus; the categorizer-api and llama-server join for `/metrics` scraping.

categorizer-api is the only container that joins all four: `[proxy, categorizer_net, miplata_db, observability]`.

### 5.4 Auth + secrets

- `.vault_pass` lives at the root of `vps-infrastructure/`, gitignored. All ansible commands pick it up via `ANSIBLE_VAULT_PASSWORD_FILE=.vault_pass`.
- `vault.yml` holds ALL secrets + identifying info (IPs, domain, ACME email). `vault.yml.example` is the redacted template.
- Traefik uses the **Aetherinox/traefik-api-token-middleware** plugin (already loaded globally by the `traefik` role's `traefik.yml.j2`) for bearer auth. Every service exposing an API wires its own `…-bearer` middleware via Docker labels. Examples: `obs-bearer` for Loki/Prometheus push, and now `categorizer-bearer` on the categorizer router.
- The categorizer specifically uses **`Authorization: Bearer <vault_categorizer_api_token>`**. The 32-byte hex token is generated once via `openssl rand -hex 32` and stored in vault.

### 5.5 Jenkins deploy-key + webhook dance (easy to forget)

Each repo Jenkins builds from needs:
1. A **GitHub deploy key** on the repo (public SSH key set in repo Settings → Deploy keys).
2. The matching **private key in vault**, e.g. `vault_jenkins_categorizer_ssh_key`.
3. A **credential in JCasC** mapping the private key to a credentialsId (`github-categorizer-ssh-key`).
4. A **`pipelineJob` in JCasC** referencing that credentialsId.
5. A **push webhook on the GitHub repo** pointing at `https://jenkins.web.<domain>/github-webhook/` — without this, Jenkins never hears about pushes.

I did (1)–(5) for miplata-categorizer during the build-out. Reusing the pattern for a future app means repeating the same 5 steps.

### 5.6 Deploy pipeline flavor

There are two distinct deploy flavors in this infra:

- **Ansible-only deploys** (e.g. Traefik, DNS, monitoring, the llm_inference role). Container images come from public registries; `ansible-playbook playbook.yml` is the full deploy. Idempotent by virtue of `pull: missing` + `state: present`.
- **App deploys via Jenkins** (miplata, miplata-categorizer, and eventually fine-tuning jobs). Jenkins builds the Docker image from the app's `Dockerfile`, pushes to the self-hosted **Zot registry** at `registry.web.<domain>`, then SSH-deploys to the main VPS via `docker compose pull && docker compose up -d`. The Ansible role for the service only provisions `/opt/<service>/`, renders `.env` + `docker-compose.yml`, pre-creates directories, and tolerates the first-bootstrap race where Zot has no image yet (`failed_when: false` on pull). Same pattern for miplata and the new categorizer.

### 5.7 Local bank-statement dataset

Sebas has real Colombian bank statements at `/home/sebastian/Documents/yo/extractos_bancarios/`:
- `bancolombia/*.xlsx` — Bancolombia savings account exports (accounts 0810 + 9855), MasterCard 1194, AmEx 1916, and a "Tapa_Resumen" file. Several months back to 2024-03.
- `nequi/*.pdf` — Nequi digital-wallet statements. **PDFs are password-protected.** The password is Sebas's cédula: `1007055144`. Use `pdftotext -upw 1007055144 -layout <file>` to extract.

The normalizer's regex patterns and the 50-tx gold set were both grounded in strings extracted from these files — *do not* guess patterns; re-read a real statement if adding new rules.

### 5.8 Colombian-specific domain knowledge

Several patterns in transaction strings only make sense with context:

| Token / pattern | Meaning |
|---|---|
| `BRE-B` | Bancolombia's low-value transfer system (like a P2P QR payment). Appears as `PAGO EN QR BRE-B: <merchant>`, `ENVIO CON BRE-B A: <name>`, `RECIBI POR BRE-B DE: <name>`. |
| `PSE` | Pagos Seguros en Línea — Colombia's inter-bank online-payment system. `COMPRA PSE EN <biller>` is a one-off bill payment (e.g. Compensar, tax authorities). |
| `GMF` / `4X1000` | Gravamen al Movimiento Financiero — 0.4% financial-transaction tax. Appears as `IMPTO GOBIERNO 4X1000` (direct debit) or `CXC IMPTO GOBIERNO 4X1000 MON` (deferred receivable). |
| `CUOTA MANEJO` | Monthly card-management fee. `TRJ DEB` = debit card; `TC` = credit card. |
| `MORA TARJETA` | Late-payment penalty on a credit card. |
| `ABONO SUCURSAL VIRTUAL` / `PAGO SUC VIRT TC MASTER PESOS` | Pays own credit card from own savings via Bancolombia's virtual branch. **These are internal transfers**, not expenses. |
| `RETIRO EN CAJERO` | ATM cash withdrawal. In this system it's classed as `movimientos_internos.retiro_efectivo` (cash is a money container for the user, not immediately "spending"). |
| `Recarga desde: COINK` | Coink is a cash-to-digital-wallet network — reload into Nequi. **Internal transfer.** |
| `Recarga Cívica` | Medellín metro's "Cívica" transit card reload. Classed as `transporte.transporte_publico`. |
| `DLO*`, `DL*`, `BOLD*`, `AMZ*`, `TST*`, `SQ*` | Payment-processor prefixes on credit-card statements. Strip them before matching the merchant. |
| `MAR*** ELI*** GON*** DIA***` | Anonymized Nequi counterparty name (bank anonymizes non-contacts). The normalizer emits `nombre_anonimo` as a sentinel token; rules fall through to retrieval / LLM. |
| `COMPRA EN ...` | Generic "purchase from" prefix — strip it. |
| `Para <NAME>` / `De <NAME>` / `ENVIO CON BRE-B A: <NAME>` / `RECIBÍ A MI LLAVE DE: <NAME>` | P2P transfer to/from a named contact. Text normalizer substitutes `envio_persona` / `recibido_persona` sentinel tokens. |

### 5.9 Sebas's real money containers (as of now)

Six own_accounts are seeded by Alembic migration `0002`:

| Slug | Display | Institution | Tail | Aliases |
|---|---|---|---|---|
| `bancolombia_ahorros_0810` | Bancolombia Ahorros 0810 | bancolombia | 0810 | bancolombia, ahorros |
| `bancolombia_ahorros_9855` | Bancolombia Ahorros 9855 | bancolombia | 9855 | bancolombia, ahorros |
| `bancolombia_mastercard_1194` | Bancolombia MasterCard 1194 | bancolombia | 1194 | mastercard, tc_master, tc |
| `bancolombia_amex_1916` | Bancolombia AmEx 1916 | bancolombia | 1916 | amex, americanexpress |
| `nequi` | Nequi | nequi | — | nequi |
| `coink_wallet` | Coink | coink | — | coink |

### 5.10 The account_slug impedance mismatch (a known compromise)

miplata's `Transaction.accountId` is a Prisma UUID. When miplata's `CategorizerHttpService` calls `/v1/categorize`, it passes that UUID as the categorizer's `account_slug`. But the categorizer's `own_accounts` table is keyed on friendly slugs (`bancolombia_ahorros_0810`, etc.), not UUIDs.

**Effect:**
- The **text-based internal-transfer rule** (`resolve_internal_transfer_by_text`) still works — it matches against the `aliases` list, which includes `bancolombia`, `nequi`, `coink`, etc. These aliases hit regardless of what `account_slug` was passed.
- The **paired-tx internal-transfer rule** (`_paired_internal_transfer` in cascade.py) does NOT work until we reconcile the UUIDs. It filters on `account_slug != tx.account_slug`, which compares UUIDs to friendly slugs. The fix is a follow-up sync script that either:
  (a) pulls miplata's `accounts` table via the already-wired `MIPLATA_RO_DATABASE_URL`, or
  (b) updates every own_account's `slug` column to the corresponding miplata UUID, or
  (c) adds the miplata UUID as an extra alias + refactors the paired-tx query to match on alias.

Design decision still to be made. Tracked in plan.md's "open questions".

### 5.11 Current runtime state

**Right now (2026-04-19, end of this session):**
- `llama-server` on VPS: **up, healthy**, Qwen3-4B-Instruct-2507 Q5_K_M loaded, serving at `http://llama-server:8080/` on `categorizer_net`. Not exposed to Traefik.
- `categorizer-api` + `categorizer-postgres`: **NOT YET DEPLOYED** — the first Jenkins `categorizer-deploy` build is in flight (triggered by an empty commit with the webhook installed). Once it lands, `alembic upgrade head` runs automatically as a pipeline stage, creating tables and seeding own_accounts + merchants.
- miplata containers: **still running the old image**. Ansible re-rendered `/opt/miplata/.env` with `CATEGORIZER_BASE_URL` + `CATEGORIZER_API_TOKEN`, but the old image doesn't have `CategorizerHttpService` — it uses the old classifier services. Since `ANTHROPIC_API_KEY` was never in env, classification is currently a no-op (same state as before this session). Miplata rebuild is pending a push to the feat branch or merge to master.

### 5.12 Memory system records for future Claude Code sessions

The user has a persistent memory file at `~/.claude/projects/-home-sebastian-versioned-code-vps-infrastructure/memory/`. Relevant entries for this project:

- `vault_password.md` — `.vault_pass` at repo root.
- `vps_topology.md` — two-host setup (main + jenkins) with IPs.
- `dns_management.md` — PowerDNS is authoritative; `pdnsutil` takes the SHORT name, not FQDN.
- `miplata_deploy_pipeline.md` — GitHub → Jenkins → Zot → SSH-deploy to VPS.
- `feedback_ambition.md` — "Don't treat primitive existing code as the gold standard when intent is to improve."
- `miplata_categorization_primitive.md` — the current Claude wrapper is a historical artifact, not a spec.
- `feedback_autonomy.md` — for "fix X / make CI pass" tasks, commit+push+iterate without asking.
- `coexisting_local_projects.md` — Sebas runs miplata + base/treasury locally; prefer non-default ports.

### 5.13 Bearer-token policy

The categorizer API token is embedded in the Traefik middleware labels at container-create time. Rotating the token therefore requires re-rendering the compose file and `docker compose up -d` to recreate the container. The same is true for any other `apitokenmiddleware` tokens in this infra — there's no hot-reload path.

### 5.14 LLM tuning decisions specific to this hardware

- `LLAMA_ARG_THREADS=5` — memory-bandwidth saturation on dual-channel DDR4 happens at ~5 threads (confirmed in llama.cpp discussion #3167). Past 5 threads, gen tok/s plateaus.
- `LLAMA_ARG_THREADS_BATCH=6` — prompt eval IS compute-bound, so we use all 6 cores for batch.
- `--cpuset "0-4"` on the container — pins 5 cores, leaves core 5 for the rest of the host (miplata, monitoring, Traefik, PowerDNS). Without this, llama-server would starve them.
- `LLAMA_ARG_CACHE_TYPE_K=q8_0 LLAMA_ARG_CACHE_TYPE_V=q8_0` — quantized KV cache saves ~400 MB at 2K context with <5% latency cost (at our context size). **Don't enable on Gemma** (Ollama issue #9683) — OK on Qwen.
- `LLAMA_ARG_CTX_SIZE=2048` — bigger is wasted budget for ~500-token system prompts + ~40-token outputs. Bigger contexts would force more KV RAM.
- `LLAMA_ARG_N_PARALLEL=2` — 2 slots for concurrent requests, useful when miplata ingests a multi-tx statement.
- `LLAMA_ARG_ENDPOINT_SLOTS=0` + `--no-slots` + `--no-webui` — locks down the /slots debug endpoint and disables the web UI. Belt-and-suspenders because the env-var behavior for `false` on llama.cpp is inconsistent across versions.

### 5.15 Taxonomy design notes

- The taxonomy lives in `config/taxonomy.yaml`. 13 top-level categories, ~54 children. Spanish labels, slugs are `parent.child`.
- The `implemented: false` marker is a **carcass** flag — the category exists in the contract so miplata and the UI can reference it, but the rule tier has no patterns for it yet and the kNN tier won't see examples for a while. Example carcasses: `educacion.*`, `compras.ropa`, `compras.electronica`, `hogar.arriendo`, etc. They will fill in naturally as corrections land.
- **`sin_clasificar.pendiente`** is the explicit "I don't know — ask the user" bucket. The cascade rejects to this when no tier meets its confidence threshold. Do NOT repurpose this category for things the system is confidently uncertain about (that's what the `/v1/uncertain` endpoint surfaces).
- **`movimientos_internos`** is a first-class branch with 5 children: `entre_bancos`, `pago_tarjeta_propia`, `retiro_efectivo`, `recarga_monedero`, `ajuste_contable`. The first 4 are actively detected; the last is a carcass for future ledger adjustments.
- English labels on miplata's existing `Category` table are being **retired**, not translated. The `Transaction.categorizerSlug` column is authoritative going forward; `categoryId` stays for backward compat during the migration.

### 5.16 The 50-tx gold set provenance

`config/gold_set_v1.jsonl` was hand-labeled from **real transaction strings** extracted from Sebas's `extractos_bancarios/` during this session. Not synthetic. Labels reflect Claude's best-effort categorization based on the taxonomy + cross-referenced patterns. Two rows are deliberate `sin_clasificar.pendiente` cases (e.g. `COMPRA PSE EN Banco` — no merchant tail, could be any PSE biller; `SEBASTIAN MENDOZA` — ambiguous self-transfer description). These exercise the reject path so the cascade's rejection behavior gets tested before real traffic arrives.

Accuracy target when seeded: the gold set + own_accounts + merchants together should give the kNN tier enough signal to autoresolve ~40% of similar incoming transactions from day 1, with rules covering another ~40% deterministically.

### 5.17 Why there's no Claude or OpenAI fallback even as an emergency escape hatch

Sebas's stated policy: "self-hosted only. I want to see how far this goes." The cascade's tier 9 (reject) is not a degraded mode — it's the *correct* mode when the local system cannot confidently categorize. Rejected rows surface via `GET /v1/uncertain` for user review; every human correction feeds back into the retrieval index (immediately) and the fine-tuning corpus (later). This is more valuable than a silent API guess because it generates labeled data for the long tail.

If this policy is ever revisited, the clean re-entry point is adding a tier 9 that calls `AsyncOpenAI(base_url="https://api.anthropic.com/v1")` (Claude supports the OpenAI-compat endpoint) behind a feature flag. The cascade trace already includes a `"reject"` tier step, so swapping it out later is a localized change.

### 5.18 Jenkins image build specifics

The `categorizer-deploy` Jenkinsfile uses `agent { docker { image 'python:3.12-slim-bookworm'; reuseNode true } }` for lint + test stages. This relies on the `docker-workflow` plugin (confirmed installed — see `roles/jenkins/files/plugins.txt`). The image stage builds the project's own `Dockerfile` (multistage: `python:3.12-slim` builder installing `pip install .` then runtime copying `/install` + source), tags as `registry.web.<domain>/miplata/categorizer:${BUILD_NUMBER}` + `:latest`, pushes both. The Deploy stage SSHes to the VPS as `ubuntu@<vault_main_vps_ip>` using the `vps-ssh-key` credential and runs `docker compose pull && docker compose up -d` in `/opt/categorizer/`. The DB Migrate stage follows with `docker compose run --rm api alembic upgrade head`.

### 5.19 Observability + scraping

The main VPS already runs Prometheus + Loki + Grafana + Alloy + cAdvisor. Prometheus scrapes services that expose `/metrics` on the `observability` Docker network (scrape config in `roles/monitoring/templates/prometheus.yml.j2`). The categorizer-api container joins `observability` and exposes `/metrics` on its main port, so adding a scrape target in `prometheus.yml.j2` is a follow-up task. Existing metrics (all bounded-cardinality): `categorizer_predictions_total{tier,status}`, `categorizer_corrections_total{parent_was_correct}`, `categorizer_latency_seconds`, `categorizer_tier_latency_seconds{tier}`, `categorizer_embedding_seconds`.

llama-server also exposes `/metrics` (via `LLAMA_ARG_ENDPOINT_METRICS=1`). Same follow-up applies.

### 5.20 How a fresh Claude Code session should resume

Assume zero memory of this session. To get oriented in under 10 minutes:

1. **Read this file + `plan.md`** (both in `miplata-categorizer/docs/`). They're the full state of the world.
2. **Check the auto-memory index** at `~/.claude/projects/-home-sebastian-versioned-code-vps-infrastructure/memory/MEMORY.md` — entries that matter here: `feedback_ambition.md`, `miplata_categorization_primitive.md`, `miplata_deploy_pipeline.md`, `vps_topology.md`, `vault_password.md`.
3. **Check branch state in all three repos** per §5.1. Do not assume anything is pushed — some commits are deliberately held local.
4. **Verify runtime state** on the VPS (don't just trust this doc):
   ```bash
   cd /home/sebastian/versioned-code/vps-infrastructure/.claude/worktrees/ai-model
   ANSIBLE_VAULT_PASSWORD_FILE=.vault_pass ansible vps -e "@vault.yml" -m shell -a \
     "docker ps | grep -E 'llama-server|categorizer' ; \
      docker exec llama-server curl -fsS http://127.0.0.1:8080/health 2>/dev/null || echo 'llama not healthy'"
   ```
   - Expect: `llama-server` healthy.
   - May or may not be running: `categorizer-api`, `categorizer-postgres` (depends on whether Jenkins's first categorizer-deploy build succeeded by now).
5. **If the Jenkins `categorizer-deploy` build has landed**, the categorizer API is live at `https://categorizer.web.vespiridion.org/healthz` (with `Authorization: Bearer <vault_categorizer_api_token>`). The Alembic `upgrade head` step runs as a Jenkinsfile stage and seeds `own_accounts` + `merchants` (migration `0002`).
6. **Seed the gold set** (once the API is up):
   ```bash
   cd /home/sebastian/versioned-code/miplata-categorizer
   ANSIBLE_VAULT_PASSWORD_FILE=../vps-infrastructure/.vault_pass \
     ansible-vault view ../vps-infrastructure/vault.yml | grep vault_categorizer_api_token
   # then:
   CATEGORIZER_BASE_URL=https://categorizer.web.vespiridion.org \
     CATEGORIZER_API_TOKEN=<token from above> \
     python scripts/seed_gold_set.py config/gold_set_v1.jsonl
   ```
7. **Check test suite still passes**:
   ```bash
   cd /home/sebastian/versioned-code/miplata-categorizer
   python3.12 -m venv .venv && .venv/bin/pip install '.[dev]'
   PYTHONPATH=src .venv/bin/python -m pytest tests/
   ```
   Expect 70 passing tests (normalize + rules).

After that, pick the next roadmap item from `plan.md` §7 (remaining work) and go.

### 5.21 Sources that shaped the design

These are the authoritative sources the current decisions trace back to. Read these before making non-trivial architectural changes — they're the upstream of the choices in this repo, not just "further reading".

1. **llama.cpp repo + GitHub Discussions** — https://github.com/ggml-org/llama.cpp
   The CPU-perf threads are where the operational knobs in `roles/llm_inference/` come from:
   - Discussion #3167 (thread-saturation on DDR) → `LLAMA_ARG_THREADS=5`.
   - Discussion #5932 (KV-cache quantization) → `--cache-type-k q8_0 --cache-type-v q8_0`.
   - Discussion #18030 (parallel batching on CPU) → `--cont-batching`, `--parallel 2`.
   - Discussion #13606 / #20574 (prompt-cache reuse) → why we keep the system prompt stable and expect KV-cache reuse to amortize its eval cost.
   - `tools/server/README.md` in the repo is the canonical reference for the JSON-schema grammar (`response_format: {type: "json_schema", json_schema: {…}}`) we use in `src/categorizer/llm.py` to force the categorization output shape.

2. **Thinking Machines — "LoRA Without Regret"** — https://thinkingmachines.ai/blog/lora/
   The 2025 consolidation of LoRA best practices. Every hyperparameter in the QLoRA plan sketched in `plan.md` (r=16, α=32, `target_modules="all-linear"`, LR 2e-4 as ~10–15× of full-FT, why attention-only LoRA underperforms even with higher rank) traces back to this post. The TRL reproduction at https://huggingface.co/docs/trl/main/en/lora_without_regret has the runnable counterpart — use it as the template when implementing the fine-tuning pipeline (carcass under `plan.md` §7.1).

3. **Berkeley Function-Calling Leaderboard (BFCL)** — https://gorilla.cs.berkeley.edu/leaderboard.html
   The reason **Qwen3-4B-Thinking-2507** is the LLM pick for tier 4 (71.2 on BFCL v3 for a sub-8B model) and why we chose Hermes-style tool parsing over Phi-4-mini (two open reliability bugs for tool calls), Gemma 3 (restrictive license + no public FC evidence), or raw Llama 3.2. Also important: BFCL v2 / v3 / v4 are **not comparable across versions** (v4 is strictly harder). When upgrading models, compare within a single BFCL version or you'll draw wrong conclusions about relative model quality.

4. **Plaid Engineering Blog — parsing + enrichment posts**
   - https://plaid.com/blog/how-plaid-parses-transaction-data/
   - https://plaid.com/blog/ai-enhanced-transaction-categorization/

   The gold standard public write-up of a production tx-categorization pipeline. Directly inspired this project's:
   - Three-tier merchant resolver (regex → dictionary → ML/LLM) — implemented as `normalize.py` → `merchant.py` seeded dict → LLM `lookup_merchant` tool.
   - The confidence-level enum idea (VERY_HIGH / HIGH / MED / LOW) as a cleaner product abstraction than a raw 0–1 float. We return the raw float in the API today, but a follow-up is to add a derived `confidence_band` field that downstream consumers use for routing decisions.
   - Using **MCC as a stable semantic anchor** alongside the user-facing taxonomy, so renames and re-orgs of the Spanish category tree don't corrupt historical data. The `merchants.mcc_hint` column exists for this; the categorizer doesn't output it today but the data's there.

**Honorable mentions** (hit often during planning, didn't make the top-4):
- **Unsloth docs** — https://unsloth.ai/docs — for the actual fine-tune recipe when we get there. Worth revisiting specifically the Qwen3 page right before starting training because version-specific QLoRA caveats land here first.
- **r/LocalLLaMA** — community CPU benchmarks that plug the gap when llama.cpp Discussions don't have your exact hardware. Search "Haswell AVX2 llama.cpp tok/s" for reference points.
- **pgvector README** — https://github.com/pgvector/pgvector — origin of the "use Postgres, skip FAISS below ~50k rows" decision, plus the `ivfflat` index tuning knobs (`lists=100`) used in migration `0001`.

### 5.22 Things that would break the system subtly

- **Changing the `embedding_model` setting** after labeled tx exist. The stored vectors are model-specific; switching would require re-embedding the corpus (simple script but not automated).
- **Reordering category enum in `taxonomy.yaml`** — not functionally harmful (slugs are still slugs), but any downstream dashboards aggregating by parent would be cleaner if parent ordering is stable.
- **Dropping the `populate_by_name=True` on `TransactionIn`** — the miplata client sends `date` (alias) and we parse as `tx_date`; removing would break request parsing.
- **Running two llama-server instances sharing the same model file** — not a problem for mmap but requires tuning cpuset so they don't step on each other. We don't do this currently.
- **Changing the Traefik-middleware bearer token via vault without re-running Ansible** — the token is labeled on the container at create time; a `docker compose up -d` is required to take effect.
