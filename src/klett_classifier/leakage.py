"""Conservative identity, provenance and approximate-semantic grouping.

A similarity detector cannot *prove* semantic non-overlap. Fail closed when
required models/index packages are missing; explicitly report residual risks.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class LeakageError(ValueError):
    pass


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def text_hash(text: str) -> str:
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, size: int):
        self.p = list(range(size))
        self.sz = [1] * size

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def provenance_graph(frame: pd.DataFrame) -> UnionFind:
    graph = UnionFind(len(frame))
    # Group on EVERY shared identity, not just a single --group-column.
    for name in ("pair_id", "group_id", "family", "source_conversation_id", "source_record_id"):
        if name not in frame.columns:
            continue
        seen = {}
        for i, value in enumerate(frame[name].tolist()):
            if pd.isna(value) or not str(value).strip():
                continue
            value = str(value).strip()
            if name == "family" and "source" in frame.columns:
                value = str(frame.iloc[i]["source"]) + "::" + value
            if value in seen:
                graph.join(i, seen[value])
            else:
                seen[value] = i
    seen = {}
    for i, value in enumerate(frame["text"].tolist()):
        key = canonical(value)
        if key in seen:
            graph.join(i, seen[key])
        else:
            seen[key] = i
    return graph


def approximate_neighbors(texts: list[str], threshold: float) -> list[tuple[int, int, float]]:
    """MinHash candidate generation, then character n-gram TF-IDF cosine verification.

    LSH is approximate (may miss pairs); intended as one of multiple barriers.
    """
    from datasketch import MinHash, MinHashLSH
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [canonical(t) for t in texts]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32)
    matrix = vectorizer.fit_transform(texts)
    lsh = MinHashLSH(threshold=0.50, num_perm=64)
    sketches = []
    for i, text in enumerate(texts):
        mh = MinHash(num_perm=64)
        grams = {text[j:j + 3] for j in range(max(len(text) - 2, 1))}
        for gram in grams:
            mh.update(gram.encode("utf-8"))
        lsh.insert(str(i), mh)
        sketches.append(mh)
    edges = []
    for i, mh in enumerate(sketches):
        for js in lsh.query(mh):
            j = int(js)
            if j <= i:
                continue
            similarity = float(matrix[i].multiply(matrix[j]).sum())
            if similarity >= threshold:
                edges.append((i, j, similarity))
    return edges


def semantic_neighbors(texts: list[str], model_name: str, threshold: float, k: int,
                       batch_size: int = 64) -> list[tuple[int, int, float]]:
    """ANN on multilingual embeddings; similarity is an imperfect proxy for intent.

    hnswlib's ANN recall is not 100%: absence of an edge is not a proof.
    """
    import hnswlib
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name)
    vectors = embedder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    if not len(vectors):
        return []
    index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), ef_construction=250, M=32, random_seed=42)
    index.add_items(vectors, np.arange(len(vectors)))
    index.set_ef(max(150, k * 3))
    indexes, distances = index.knn_query(vectors, k=min(k + 1, len(vectors)))
    found = {}
    for i, (row, dist) in enumerate(zip(indexes, distances)):
        for j, d in zip(row, dist):
            j = int(j)
            if i == j:
                continue
            score = float(1.0 - d)
            if score >= threshold:
                key = (min(i, j), max(i, j))
                found[key] = max(score, found.get(key, -1.0))
    return [(i, j, score) for (i, j), score in sorted(found.items())]


@dataclass(frozen=True)
class GroupingSettings:
    near_threshold: float = 0.88
    semantic_threshold: float = 0.91
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    semantic_k: int = 40
    max_component_fraction: float = 0.20


def build_leakage_groups(frame: pd.DataFrame, settings: GroupingSettings,
                         near_fn=approximate_neighbors,
                         semantic_fn=semantic_neighbors) -> tuple[pd.DataFrame, dict]:
    """Union all same-label links. Cross-label near matches are review blockers.

    Same-template groups across different classes are allowed in the same split,
    but a high-similarity cross-label pair must be reviewed first.
    """
    frame = frame.reset_index(drop=True).copy()
    # A bilingual pair represents one original intent. Contradictory labels
    # in the same pair indicate corrupt training data, never valid contrast.
    if "pair_id" in frame:
        if frame.groupby("pair_id")["label"].nunique().gt(1).any():
            raise LeakageError("A pair_id contains contradictory labels")
    uf = provenance_graph(frame)
    texts = frame["text"].tolist()
    labels = frame["label"].tolist()
    conflicts = []
    counters = {}

    def check_size(stage: str) -> None:
        # Run BEFORE expensive semantic embeddings, not only after they finish.
        sizes = Counter(uf.find(i) for i in range(len(frame)))
        largest = max(sizes.values())
        if largest > len(frame) * settings.max_component_fraction:
            raise LeakageError(
                f"Oversized related-text component after {stage}: "
                f"{largest}/{len(frame)} rows ({largest / len(frame):.1%}); "
                f"limit {settings.max_component_fraction:.0%}. "
                "This can be caused by transitive links between synthetic templates. "
                "Inspect source/family and similarity thresholds; "
                "--skip-semantic avoids the embedding stage, and --provenance-only "
                "keeps pair/family/exact-text grouping without similarity checks. "
                "These options weaken leakage detection."
            )

    check_size("provenance")
    # Evaluate stages one at a time. Previously both neighbor functions were
    # evaluated eagerly, so a lexical failure still spent ~1h embedding texts.
    for kind, pairs in (
        ("lexical", near_fn(texts, settings.near_threshold)),
    ):
        counters[kind + "_candidate_edges"] = len(pairs)
        for i, j, sim in pairs:
            uf.join(i, j)
            if labels[i] != labels[j]:
                conflicts.append({"type": kind, "fingerprints": sorted((text_hash(texts[i]), text_hash(texts[j]))),
                                  "similarity": round(float(sim), 4), "label_a": labels[i],
                                  "label_b": labels[j], "text_a": texts[i], "text_b": texts[j]})
        check_size(kind)

    pairs = semantic_fn(texts, settings.semantic_model,
                        settings.semantic_threshold, settings.semantic_k)
    counters["semantic_candidate_edges"] = len(pairs)
    for i, j, sim in pairs:
        uf.join(i, j)
        if labels[i] != labels[j]:
            conflicts.append({"type": "semantic", "fingerprints": sorted((text_hash(texts[i]), text_hash(texts[j]))),
                              "similarity": round(float(sim), 4), "label_a": labels[i],
                              "label_b": labels[j], "text_a": texts[i], "text_b": texts[j]})
    check_size("semantic")
    roots = [uf.find(i) for i in range(len(frame))]
    # Semantic+identity connected components. Stable hash of lexicographically smallest text.
    by_root = defaultdict(list)
    for i, root in enumerate(roots):
        by_root[root].append(i)
    id_by_root = {r: text_hash(min(texts[i] for i in members))
                  for r, members in by_root.items()}
    frame["leakage_group"] = [id_by_root[r] for r in roots]
    report = {**counters, "rows": len(frame), "groups": len(by_root),
              "largest_group": max(map(len, by_root.values())),
              "cross_label_near_matches": conflicts,
              "cross_label_near_match_count": len(conflicts),
              "semantic_model": settings.semantic_model,
              "semantic_threshold": settings.semantic_threshold,
              "approximate_methods": True}
    if conflicts:
        # Cross-class contrastive examples may be legitimately similar. Manual review
        # is required; do not silently discard them or relabel them.
        return frame, report
    return frame, report


def validate_no_shared_components(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, a, b in (("train/validation", train, val), ("train/test", train, test),
                       ("validation/test", val, test)):
        overlap = set(a["leakage_group"]) & set(b["leakage_group"])
        if overlap:
            raise LeakageError(f"Leakage groups overlap in {name}: {len(overlap)}")
        overlap = set(a["text"].map(canonical)) & set(b["text"].map(canonical))
        if overlap:
            raise LeakageError(f"Normalized texts overlap in {name}: {len(overlap)}")
