# KlettClassifier — secure EN/BG intent classification

**Python 3.13 · XLM-RoBERTa-base · FastAPI · auditable data split · tests**

This repository implements the **Klett Bulgaria AI Software Developer exercise**: given a text query, assign exactly one of:

- `Технически проблем`
- `Достъп или регистрация`
- `Учебно съдържание`
- `Друго`

It accepts **Bulgarian or English**. The model needs **no translation at inference**. The trained model is not included because it must be produced from appropriately licensed and reviewed data; for instant demonstration a deterministic **rules backend** is included. The rules backend has **no calibrated confidence** and does **not** represent XLM-R accuracy. See [SECURITY.md](SECURITY.md) and [DATA.md](DATA.md).

## Quick start — Windows PowerShell and Python 3.13

Run every command from the folder that contains `pyproject.toml`:

```powershell
py -V:3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\klett.exe predict --backend rules --text "Видеоурокът по математика не се зарежда."
.\.venv\Scripts\klett.exe interactive --backend rules
```

The CLI returns JSON with `category`, `confidence`, `scores`, and `method`. In rules mode `confidence` is `null` (not a probability). Empty or malformed text returns a clear error; the process exits nonzero for invalid one-shot input.

## Secured local HTTP demonstration

Set a **unique secret** (avoid saving it in source control):

```powershell
$env:KLETT_API_KEY = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
.\.venv\Scripts\klett.exe serve --backend rules --host 127.0.0.1 --port 8000
```

In a *second* PowerShell terminal, set the **same** secret in `$env:KLETT_API_KEY` (environment values are per process), then:

```powershell
$headers = @{ Authorization = "Bearer $env:KLETT_API_KEY" }
$body = @{ text = "Видеоурокът по математика не се зарежда." } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/predict -Method POST `
  -Headers $headers -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

Endpoints: `POST /v1/predict`, `POST /v1/predict/batch` (max 32 items), `GET /health/live` (public), `GET /health/ready` (authenticated). Interactive Swagger is disabled intentionally on the service. No DB or external AI API is required.

### API example

**Request**: `{"text":"Видеоурокът по математика не се зарежда."}`

**Representative rules response** (exact values, no inference model required):

```json
{"category":"Технически проблем","confidence":null,"scores":{},"method":"rules"}
```

Model responses contain softmax `confidence`, per-class `scores`, and `method: "xlm-roberta"`. These softmax values are **not calibrated probabilities**.

## Training one bilingual XLM-RoBERTa model

Full dataset workflow needs optional libraries including `hnswlib`, which may require **Microsoft C++ Build Tools** on Windows. If this extra fails to install, do not skip the leakage audit and do not train on mixed EN/BG using a random row split.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,leakage]"
.\.venv\Scripts\python.exe .\datasets_processing\build_dataset.py --target-per-class 3500 --strict --output data\english_train_candidates.csv
.\.venv\Scripts\python.exe .\datasets_processing\build_multilingual_dataset.py --english-input data\english_train_candidates.csv --output-dir data\multilingual
.\.venv\Scripts\klett.exe split-data --data data\multilingual\multilingual_with_pair_ids.csv --output data\splits
```

The split uses provenance (`pair_id`, `source`, `family`), exact/lexical matches, and approximate multilingual semantic neighbors. If it discovers close texts with **different labels**, it **stops** and writes `data/splits/review_conflicts.json` for human review. Acknowledging genuinely different-intent contrasts is **not** a request to ignore them: approved pairs are always kept within the same split. For each genuinely intentional contrast copy only its `fingerprints` field into `approved.json` and add `"decision":"intentional_contrast"`:

```json
[{"fingerprints":["<sha256 text A>","<sha256 text B>"],"decision":"intentional_contrast"}]
```

Replace the placeholders with the real 64-character hashes from `review_conflicts.json`, and rerun:

```powershell
.\.venv\Scripts\klett.exe split-data --data data\multilingual\multilingual_with_pair_ids.csv --output data\splits --approved-review approved.json
.\.venv\Scripts\klett.exe train --split-dir data\splits --output models\classifier-en-bg --epochs 5 --batch-size 8
.\.venv\Scripts\klett.exe evaluate --data data\splits\test.csv --model models\classifier-en-bg --report models\classifier-en-bg\test_report.json
.\.venv\Scripts\klett.exe interactive --model models\classifier-en-bg
```

A *real* test set of original EN/BG customer queries, manually annotated and not generated from the same templates, is still necessary to establish generalization and any **Macro-F1 ≥ 0.95** claim. Neither a translated split nor an artificially template-heavy holdout is sufficient evidence. Detailed notes in [DATA.md](DATA.md).

## Fast workaround for large synthetic components (v3.0.1 hotfix)

If `split-data` stops with `Oversized related-text component` after spending a long
period embedding the corpus, **do not run the same command again unchanged**.
The error means relatedness links connect more than 20% of all rows into one group.
This is common with heavily templated synthetic data, especially because
similarity edges are transitively connected. It is not a Hugging Face download error.

For the **fastest demo** on Windows, use **provenance-only** splitting. It does not
need `hnswlib`, `sentence-transformers` or `datasketch` and does not download
1.11 GB of semantic weights:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data]"
.\.venv\Scripts\klett.exe split-data --data data\multilingual\multilingual_with_pair_ids.csv --output data\splits --provenance-only
```

This **still groups** `pair_id`, `source + family`, source record/conversation
IDs (if present), and exact normalized duplicates. However, **near-duplicate
and semantic paraphrase checks are disabled**, so the resulting test metrics
must not be advertised as leakage-free or production-validated.

A middle option preserves lexical near-duplicate detection without running the
slow embedding model:

```powershell
.\.venv\Scripts\python.exe -m pip install "datasketch>=1.6,<2"
.\.venv\Scripts\klett.exe split-data --data data\multilingual\multilingual_with_pair_ids.csv --output data\splits --skip-semantic
```

If the lexical pass itself makes a huge connected group, the command will now
stop **before** loading semantic embeddings, with a diagnostic naming the stage
and group size. If provenance-only fails too, the source/family IDs are too broad;
fix the data rather than bypassing grouping. For a credible generalization
claim, evaluate on a separate, human-labeled dataset of natural queries.

The `split_manifest.json` explicitly records which similarity checks were
active. Do not compare provenance-only validation F1 directly with strict
semantic-audited F1 as if the evaluation protocols were equivalent.

## Security and operating constraints

- API refuses to start without `KLETT_API_KEY` (minimum 24 characters), requires constant-time Bearer comparison, disables docs, and doesn't log query text in application code.
- Limits: 32 KiB HTTP body, 4,000 characters/query, max 32 items/batch, 60 requests/minute/client IP, and 2 concurrent inferences per process; JSON content-type required. Resource saturation gives 429/503.
- Text is **data**, never executed, interpreted as instructions, or passed to a shell, database, or external tool. Unsupported controls and bidirectional overrides are rejected.
- A trained model is loaded exclusively from **local safetensors** with `local_files_only=True`, `trust_remote_code=False`; missing/corrupt model prevents model backend startup rather than silently switching to rules.
- Default HTTP binds only to `127.0.0.1`; public deployment requires a TLS reverse proxy, network controls, a gateway/centralized rate limiter, secrets management, `KLETT_ALLOW_REMOTE_BIND=true` and explicit `KLETT_ALLOWED_HOSTS`. In-process limiting is **not globally distributed**.
- Use a single service worker per GPU and monitor throughput/latency/OOM. Prefer application-level horizontal replication; a pod should restart on readiness failure.
- The package ships **no model weights, private customer data, or secrets**. Treat external datasets and model downloads as a supply-chain concern. Review licenses and consent.

See [SECURITY.md](SECURITY.md) for threat model and production rollout checklist.

## Project layout

```text
src/klett_classifier/
  api.py            FastAPI endpoints and guarded inference capacity
  security.py       Authentication, rate/body limits, unicode hygiene
  rules.py          Working deterministic fallback for demonstration
  inference.py      Bilingual XLM-R model; safe local loading
  data.py           CSV validation + split-bound test validation
  leakage.py        Provenance/lexical/semantic relatedness graph
  splitting.py      Audited group train/validation/test + SHA256 manifest
  training.py       Fine-tuning and validation Macro-F1
  evaluation.py     Independent test reporting by language
  labels.py         Four business categories
  cli.py            predict/interactive/serve/split-data/train/evaluate

datasets_processing/
  build_dataset.py  English data acquisition + controlled synthesis
  build_multilingual_dataset.py EN→BG translation, cache, bilingual IDs
tests/              Unit, API, security, leakage and tamper regression tests
```

## Integration and scaling

Call the API from the existing educational platform's backend after authentication. Use the returned category to route a ticket to the appropriate support team. On increased load add bounded queues and separate inference replicas, with request deadlines at the gateway, metrics (request rate, latency, category distribution, error rate) **without raw PII**, and a human-review route for uncertain predictions. No external AI dependency is needed at runtime: a missing trained model makes readiness fail; rules are available only if the operator explicitly selects the rules backend.

**Two natural improvements**: (1) feedback-based human relabelling/active learning and drift monitoring; (2) an abstain/manual-triage option calibrated on real held-out data (while keeping four labels for the current assignment).

## Verification and compatibility

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests datasets_processing
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Automated tests use mock Transformers weights for model-inference unit tests. The actual pretrained model download, large-scale semantic audit, training on 14k–60k rows, Windows GPU/CUDA compatibility and any target Macro-F1 require **end-to-end validation on your machine**. See `VERIFICATION.md` for evidence and limitations. Dependencies are constrained in `pyproject.toml`; for reproducible deployment generate, review and commit a platform-appropriate lockfile with `uv lock` using access to PyPI.

### Deadline / scope

The source brief explicitly says that UI, database and a fully completed production environment are **not required**. This package contains hardened foundations and a working demo, rather than a claim that a model has already been trained, security-audited externally, or certified to meet a target F1.
