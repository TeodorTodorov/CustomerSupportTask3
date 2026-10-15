# Security design and deployment controls

## Threat model

| Threat | Mitigation in this repository | Remaining risk / required deployment control |
| --- | --- | --- |
| Anonymous or guessing API clients | Mandatory secret, Bearer token constant-time compare, 401 | TLS gateway, key rotation, external secret store, audit access |
| Flooding / oversized requests | 32KiB ASGI limit even for chunked requests; 4k chars; batch 32; bounded in-process per-IP rate and concurrency | Reverse proxy request deadlines, rate limits shared across replicas, load tests |
| Injection prompt such as `ignore instructions`, SQL or shell payload | Encoder-only classifier, all input treated strictly as plain text; no tools, SQL, eval, shell, or remote calls | Adversarial text can still lower classification quality; test with realistic attacks |
| Unicode control/bidi spoofing | Reject NUL, special bidi overrides and non-whitespace C0 control characters | Does not detect every homoglyph or zero-width character |
| Unsafe model deserialization / untrusted code | Local-only, safetensors-only, `trust_remote_code=False`, explicit category mapping check | Trusted artifact registry, hash/signature verification, restricted filesystem permissions |
| Model availability | Startup fails closed for missing model; explicit `--backend rules` for demonstration/fallback | Cluster readiness probes and incident response |
| PII exposure | Avoid logging input query or tokens in app code; errors are generic | Ingress/server/access log redaction, retention policy, GDPR DPIA if applicable |
| Training-data poisoning | Preserve source/license/family/pair metadata; reject conflicts; human review of ambiguous intents | Validate external dataset labels and translation quality; provenance can itself be falsified |
| Leakage/inflated F1 | Group originals and translations, template families, exact/lexical/ANN neighbors; manifest hashes; quarantine ambiguous pairs | ANN misses are possible; require independent natural test set and human audit |
| CSV formula injection | Training reads CSV as data, never evaluates cell content | When exporting to spreadsheet apps sanitize dangerous `= + - @` prefixes *in presentation copies*, not training originals |

## Requirements before internet-facing deployment

1. Restrict port to a trusted private network. Terminate HTTPS at a reverse proxy and enable application-to-proxy firewall rules. Never expose plain HTTP over the internet.
2. Set a unique `KLETT_API_KEY` through a secrets manager, rotate regularly, and disable debug. The key is not a user-authentication system; the caller backend must authenticate end users.
3. Apply a shared, distributed rate limiter at the gateway. In-memory limiter is per process and is unsuitable as the only protection behind multiple replicas.
4. Configure proxy limits (32 KiB body, connection/read deadlines, slow-client throttling, active request limits), network egress restrictions, observability and autoscaling.
5. Run with minimal filesystem permissions, non-root identity, a read-only model mount and trusted immutable signed artifacts. Verify artifact SHA-256 externally before promotion.
6. Carry out SAST, dependency vulnerability scanning, adversarial ML testing, penetration testing and representative load testing.
7. Establish a documented PII retention policy and avoid sending support messages to unapproved third-party model/translation APIs.

## Model behavior

- Softmax is not calibrated probability. Define abstention/escalation using a held-out calibration set, not a guessed confidence threshold.
- `Друго` is a forced fourth category, not an open-world OOD guarantee; do not claim that every out-of-scope query is handled correctly.
- The API intentionally does not publish OpenAPI/Swagger to anonymous callers. Do not treat that as authentication or a substitute for a firewall.
