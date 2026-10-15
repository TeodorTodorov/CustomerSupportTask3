# Verification report (generated 2026-10-14)

Environment: Python 3.13.5, Linux, PyTorch and scikit-learn available. Hugging Face Transformers, `sentence-transformers` and `hnswlib` not installed in this evaluation runtime; network access to PyPI unavailable.

## Checks actually executed

- `python -m compileall -q src tests datasets_processing` — **PASS**.
- `PYTHONPATH=src pytest -q` — **31 passed**. Coverage of security tests includes authentication failure, bad inputs, token rejection, Unicode control bypass, >32KiB request even without Content-Length, rate limiting, inert injection text, input size and batch limits, SHA256 dataset tampering, EN/BG pair preservation, grouping paraphrases, human-reviewed contrast pairs and mock HF Trainer integration.
- `python -m pip wheel --no-build-isolation --no-deps .` — **PASS**: `klett_classifier-3.0.0-py3-none-any.whl`. Wheel contains `api.py`, `security.py`, `rules.py`, `training.py` and other modules.
- Rule backend CLI for Bulgarian technical/educational intent and English account intent — **PASS** and structured JSON produced.
- `build_dataset.py --offline --target-per-class 30` — **PASS**; 120 unique English examples, 30 per category, and a provenance report.

## NOT tested / not claimed

- Actual XLM-RoBERTa or Helsinki neural model download and model quality / target F1.
- Windows 3.13 + CUDA training, Hugging Face trainer running real checkpoints, full EN/BG translation and `hnswlib`/sentence embedding audit on 30k–60k rows.
- Public network deployment, HTTPS, shared rate limiting, high-concurrency soak and penetration tests.
- A complete dependency lock/resolved supply-chain audit, independent natural ticket annotations, or licensed model/data approvals.

The project is a hardened, reproducible-in-source implementation **with an immediately functioning rules prototype**, a tested ML training/evaluation orchestration, and explicit deployment gates. It is **not** a claim that an untrained model is production-ready, that a 0.95 F1 has been measured, or that any algorithm can guarantee detecting all semantic paraphrases.

## v3.0.1 hotfix verification

- Regression tests cover preflight oversized provenance/lexical components,
  no semantic-model load after an earlier failure, and both CLI skip flags.
- `pytest -q` executed with `PYTHONPATH=src`: **35 passed** in the supplied
  container environment. This does not establish full Windows GPU training.
- Provenance-only `split-data` smoke-tested with 320 synthetic EN/BG rows:
  256 train, 32 validation, 32 test, followed by manifest verification.
- No 1.11 GB semantic model weights or fully trained XLM-R classifier bundled.
