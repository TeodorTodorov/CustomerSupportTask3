# Dataset provenance, EN→BG and leakage audit

1. Acquire English data only from approved/licensed datasets (`Bitext`, `TriageIQ`) and controlled synthetic seed templates. Retain `text,label,source,family` for each row. Do not claim a synthetic text is a real customer ticket.
2. Translate with local Helsinki EN→BG. Cache keyed by source and translation model. Export a review file; correct invalid, untranslated, offensive, misleading or ambiguous translations before training. Bulgarian examples are **translations**, not independent source observations.
3. Every EN/BG pair shares `pair_id`. Templates share `source + family`. Source conversation IDs can also group related tickets.
4. `klett split-data` creates a relatedness graph connecting provenance links, canonical exact matches, MinHash/character TF-IDF matches and approximate multilingual sentence embedding neighbors. Related items are assigned together to **train/validation/test**; no row-wise random split.
5. Near matches with *different* classes are deliberately blocked until each pair is reviewed as an intentional contrast. Review entries use SHA256 fingerprints so they don't depend on row numbers; approved cross-class edges still land in the same split.
6. The saved `split_manifest.json` records dataset identities, model/settings, train/validation/test membership, file SHA256 and class/language distributions. Training fails if any split bytes change. The model stores a record of expected held-out test hash so `evaluate` rejects a tampered or wrong test file.
7. The graph is based on *approximate* semantic similarity. It cannot detect all meaning-preserving paraphrases or data copied in other formats. There is no universal algorithm guaranteeing zero semantic leakage.
8. For credible Macro-F1 >= 0.95, use a **separate manually labeled natural ticket set** that isn't a translation or paraphrase of any training template; evaluate EN and BG separately. Freeze it before model/hyperparameter selection. Check precision/recall per class and bootstrap confidence intervals if the sample is small.

**Critical labeling convention**: `Account` overrides a generic technical failure if the failure is about login, registration, password recovery or profile settings. `Technical` overrides `Educational content` when the intent is troubleshooting a malfunction of an educational resource. `Educational content` refers to finding or asking for learning materials. `Other` covers unrelated requests including billing and general support. This convention should be checked in the source dataset mappings.

**Failure to generate independent splits:** if one family or semantic component is too large, the split process stops. Do not lower grouping thresholds only to obtain an impressive score. Reduce template repetition, add independently authored examples, and review source labels.

## v3.0.1 operational note

`--skip-semantic` disables multilingual embedding neighbors but preserves exact,
provenance and lexical links. `--provenance-only` disables both lexical and semantic
neighbors while keeping `pair_id`, `source + family`, source record/conversation IDs
and exact normalized text grouping. These modes are **not equivalent** to the full
audit; the manifest records the enabled checks.

The splitter now checks connected-component sizes after provenance and lexical
grouping *before* executing slow semantic inference. The 20% component limit
remains a conservative guard; a giant group is a data/graph problem, not an
invitation to increase the limit without investigating.
