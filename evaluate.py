"""
CAR Evaluator — score any retriever on FinSuperQA.

One-command usage:
  python3 evaluate.py --retriever bm25 --split test
  python3 evaluate.py --retriever temporal_two_stage --k 5

Custom retriever:
  Implement a Retriever class with:
    def retrieve(query: str, corpus: list[dict], k: int) -> list[str]
  Then pass it via --retriever_path.

Output: leaderboard-ready JSON with TCA@k, mean_rank, MRR.

Supports:
  --retriever       bm25 | temporal_two_stage | two_stage | dense:<model_id>
  --retriever_path  path to Python file defining a custom Retriever class
  --split           train | dev | test | all  (default: all)
  --k               top-k cutoff (default 5)
  --n_distractors   distractor events per example (default 10, matches paper)
  --output          path for JSON output (default: stdout)

Corpus note:
  The default corpus (--n_distractors 10) uses synthetically generated
  entity-disjoint distractor events matching the paper's 12,250-event setup.
  --retriever two_stage reproduces the paper's reference TCA@5 = 0.978
  (entity probe reaches 1.000 but requires oracle NER; two_stage uses BM25
  Stage-1 with a 2.9% miss rate).

Splits:
  --split train|dev|test uses a stratified 80/10/10 split by hop type
  (800/100/100 examples). Split ID files are in data/finsuperqa_*_ids.txt.

Repository:
  https://github.com/andremir/car-retrieval
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DISTRACTOR_EVENT_TYPES = [
    "pre_clearance_approved", "pre_clearance_denied",
    "policy_acknowledged", "conflict_disclosed",
]
_DISTRACTOR_TEMPLATES = [
    "{emp} received pre-clearance to purchase/sell {ticker} on {date}.",
    "{emp}'s pre-clearance request for {ticker} was denied on {date}.",
    "{emp} acknowledged the insider trading policy for {ticker} on {date}.",
    "{emp} disclosed a potential conflict of interest regarding {ticker} on {date}.",
]
_DATES = [f"January {d:02d}, 2024" for d in range(1, 29)]


def _generate_distractors(
    examples: list[dict],
    n_per_example: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Generate entity-disjoint distractor events (no real (employee, ticker) pair)."""
    rng = random.Random(seed)
    gold_pairs: set[tuple] = {(ex["employee"], ex.get("ticker", "")) for ex in examples}
    all_emps = sorted({ex["employee"] for ex in examples})
    all_tickers = sorted({ex.get("ticker", "") for ex in examples})

    # Pre-compute all non-gold pairs to sample from
    non_gold_pairs = [(e, t) for e in all_emps for t in all_tickers
                      if (e, t) not in gold_pairs]

    distractors = []
    target = len(examples) * n_per_example
    for i in range(target):
        emp, ticker = rng.choice(non_gold_pairs)
        etype = rng.choice(_DISTRACTOR_EVENT_TYPES)
        tmpl = rng.choice(_DISTRACTOR_TEMPLATES)
        date = rng.choice(_DATES)
        text = tmpl.format(emp=emp, ticker=ticker, date=date)
        did = f"distractor_{i:06d}"
        distractors.append({
            "doc_id":     did,
            "text":       text,
            "event_type": etype,
            "entities":   {"employee": emp, "ticker": ticker},
            "timestamp":  f"2024-01-{rng.randint(1, 28):02d}T09:00:00",
        })
    return distractors


def load_finsuperqa(split: str = "all",
                    n_distractors: int = 10) -> tuple[list[dict], list[dict]]:
    """Load FinSuperQA examples and the distractor corpus.

    Returns:
      examples: list of {query, gold_doc_id, kb, hop_type, ...}
      corpus:   list of {doc_id, text, event_type, entities}
    """
    data_path = ROOT / "data" / "finsuperqa_v1.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}\n"
                                f"Clone the repo: https://github.com/andremir/car-retrieval")
    examples = []
    for line in open(data_path):
        if line.strip():
            examples.append(json.loads(line))

    if split != "all":
        split_path = ROOT / "data" / f"finsuperqa_{split}_ids.txt"
        if split_path.exists():
            ids = set(open(split_path).read().split())
            examples = [e for e in examples if e["example_id"] in ids]

    # Build flat corpus from all KB events; use event_id as the stable doc key.
    # Attach employee/ticker entities from the example level to each event.
    corpus_map: dict[str, dict] = {}
    for ex in examples:
        emp = ex.get("employee", "")
        ticker = ex.get("ticker", "")
        for event in ex["kb"]:
            did = event.get("event_id") or event.get("doc_id") or f"{ex['example_id']}_{event['event_type']}"
            if did not in corpus_map:
                corpus_map[did] = {
                    "doc_id":     did,
                    "text":       event["text"],
                    "event_type": event["event_type"],
                    "entities":   {"employee": emp, "ticker": ticker},
                    "timestamp":  event.get("timestamp", ""),
                }

    # Attach gold_doc_id to each example: last hop in provenance chain
    for ex in examples:
        prov = ex.get("provenance", [])
        if prov:
            ex["gold_doc_id"] = prov[-1]["event_id"]

    corpus = list(corpus_map.values())
    if n_distractors > 0:
        corpus.extend(_generate_distractors(examples, n_per_example=n_distractors))
    return examples, corpus


# ---------------------------------------------------------------------------
# Retriever protocol
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    def retrieve(self, query: str, corpus: list[dict], k: int) -> list[str]:
        """Return list of doc_ids in ranked order (best first)."""
        ...


# ---------------------------------------------------------------------------
# Built-in retrievers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25Retriever:
    name = "bm25"

    def __init__(self, corpus: list[dict]):
        self._corpus = corpus
        self._docs = [_tokenize(d["text"]) for d in corpus]
        n = len(self._docs)
        avgdl = sum(len(d) for d in self._docs) / max(n, 1)
        df = Counter(t for doc in self._docs for t in set(doc))
        self._idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}
        self._avgdl = avgdl
        self._n = n
        self._k1, self._b = 1.5, 0.75
        self._dl = [len(d) for d in self._docs]
        # Pre-compute per-doc term frequencies stored as plain dicts
        self._tf = [dict(Counter(d)) for d in self._docs]
        # Inverted index: term → list of doc indices that contain it
        self._inv: dict[str, list[int]] = {}
        for i, doc in enumerate(self._docs):
            for t in set(doc):
                self._inv.setdefault(t, []).append(i)

    def retrieve(self, query: str, corpus: list[dict], k: int) -> list[str]:
        qtoks = set(_tokenize(query))
        # Only score docs that contain at least one query term
        candidate_ids: set[int] = set()
        for t in qtoks:
            candidate_ids.update(self._inv.get(t, []))
        scores = []
        k1, b, avgdl = self._k1, self._b, self._avgdl
        for i in candidate_ids:
            dl = self._dl[i]
            tf = self._tf[i]
            norm = 1 - b + b * dl / avgdl
            s = sum(
                self._idf.get(t, 0) * tf.get(t, 0) * (k1 + 1) /
                (tf.get(t, 0) + k1 * norm)
                for t in qtoks
            )
            scores.append((s, i))
        scores.sort(reverse=True)
        return [corpus[i]["doc_id"] for _, i in scores[:k]]


class DenseRetriever:
    name: str

    def __init__(self, model_id: str, corpus: list[dict], batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.name = f"dense:{model_id}"
        self._model = SentenceTransformer(model_id)
        self._corpus = corpus
        texts = [d["text"] for d in corpus]
        needs_prefix = "e5" in model_id.lower()
        if needs_prefix:
            texts = ["passage: " + t for t in texts]
        self._embs = self._model.encode(texts, batch_size=batch_size,
                                        normalize_embeddings=True, show_progress_bar=True)
        self._np = np

    def retrieve(self, query: str, corpus: list[dict], k: int) -> list[str]:
        needs_prefix = "e5" in self.name.lower()
        q = "query: " + query if needs_prefix else query
        q_emb = self._model.encode([q], normalize_embeddings=True)[0]
        scores = self._embs @ q_emb
        ranking = self._np.argsort(-scores)
        return [corpus[i]["doc_id"] for i in ranking[:k]]


class TwoStageRetriever:
    """BM25 Stage-1 + entity-indexed Stage-2 (the reference architecture)."""
    name = "two_stage_bm25"

    def __init__(self, corpus: list[dict],
                 anchor_type: str = "pre_clearance_approved",
                 superseding_type: str = "blackout_announced",
                 scope_keys: tuple = ("employee_id", "security_ticker")):
        self._bm25 = BM25Retriever(corpus)
        self._corpus = corpus
        self._anchor_type = anchor_type
        self._superseding_type = superseding_type
        self._scope_keys = scope_keys
        # Build entity index: scope_key_values → list of doc_ids
        self._idx: dict[tuple, list[str]] = {}
        for doc in corpus:
            if doc["event_type"] == superseding_type:
                key = tuple(str(doc["entities"].get(k, "")).lower()
                            for k in scope_keys)
                if all(key):
                    self._idx.setdefault(key, []).append(doc["doc_id"])

    def retrieve(self, query: str, corpus: list[dict], k: int,
                 k1: int = 5) -> list[str]:
        stage1 = self._bm25.retrieve(query, corpus, k1)
        promoted, seen = [], set()
        doc_by_id = {d["doc_id"]: d for d in corpus}
        for did in stage1:
            doc = doc_by_id.get(did)
            if doc and doc["event_type"] == self._anchor_type:
                scope_key = tuple(str(doc["entities"].get(k, "")).lower()
                                  for k in self._scope_keys)
                for cand_id in self._idx.get(scope_key, []):
                    if cand_id not in seen:
                        promoted.append(cand_id)
                        seen.add(cand_id)
        remaining = [d for d in stage1 if d not in seen]
        # Add non-stage1 docs in BM25 order
        # Rank only enough extra docs to fill positions beyond promoted+remaining.
        # The caller evaluates at k ≤ 100 (evaluate() uses max(k, 100)); ranking
        # the full corpus is unnecessary and expensive at scale.
        extra_needed = max(k, 100) + len(promoted) + len(remaining)
        all_ranked = self._bm25.retrieve(query, corpus, extra_needed)
        extra = [d for d in all_ranked if d not in seen and d not in set(remaining)]
        return promoted + remaining + extra


class TemporalTwoStageRetriever:
    """BM25 Stage-1 + temporal entity index Stage-2 for FinSuperQA.

    Controlling authority = the most recent compliance event for the same
    (employee, ticker) scope.  Stage-1 retrieves any event for the entity
    pair; Stage-2 promotes the latest-timestamped event for that pair.
    Works across all hop depths (T0–T3) without knowing the event-type chain.
    """
    name = "temporal_two_stage"

    def __init__(self, corpus: list[dict],
                 scope_keys: tuple = ("employee", "ticker")):
        self._bm25 = BM25Retriever(corpus)
        self._corpus = corpus
        self._scope_keys = scope_keys
        # Build entity index: scope_key_values → doc_id of LATEST event (by timestamp)
        latest: dict[tuple, tuple] = {}  # key → (timestamp_str, doc_id)
        for doc in corpus:
            key = tuple(str(doc["entities"].get(k, "")).lower() for k in scope_keys)
            if not all(key):
                continue
            ts = doc.get("timestamp", "")
            if key not in latest or ts > latest[key][0]:
                latest[key] = (ts, doc["doc_id"])
        self._latest: dict[tuple, str] = {k: v[1] for k, v in latest.items()}

    def retrieve(self, query: str, corpus: list[dict], k: int,
                 k1: int = 5) -> list[str]:
        stage1 = self._bm25.retrieve(query, corpus, k1)
        promoted, seen = [], set()
        doc_by_id = {d["doc_id"]: d for d in corpus}
        for did in stage1:
            doc = doc_by_id.get(did)
            if not doc:
                continue
            key = tuple(str(doc["entities"].get(sk, "")).lower() for sk in self._scope_keys)
            cand_id = self._latest.get(key)
            if cand_id and cand_id not in seen:
                promoted.append(cand_id)
                seen.add(cand_id)
        remaining = [d for d in stage1 if d not in seen]
        # Rank only enough extra docs to fill positions beyond promoted+remaining.
        # The caller evaluates at k ≤ 100 (evaluate() uses max(k, 100)); ranking
        # the full corpus is unnecessary and expensive at scale.
        extra_needed = max(k, 100) + len(promoted) + len(remaining)
        all_ranked = self._bm25.retrieve(query, corpus, extra_needed)
        extra = [d for d in all_ranked if d not in seen and d not in set(remaining)]
        return promoted + remaining + extra


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(retriever, examples: list[dict], corpus: list[dict],
             k: int = 5) -> dict:
    hits, ranks = 0, []
    for ex in examples:
        gold_id = ex.get("gold_doc_id") or ex.get("answer_doc_id")
        if not gold_id:
            continue
        ranking = retriever.retrieve(ex["query"], corpus, k=max(k, 100))
        try:
            rank = ranking.index(gold_id) + 1
        except ValueError:
            rank = len(corpus) + 1
        ranks.append(rank)
        if rank <= k:
            hits += 1
    n = len(ranks)
    tca_k = hits / n if n else 0.0
    mean_rank = sum(ranks) / n if n else float("inf")
    mrr = sum(1.0 / r for r in ranks) / n if n else 0.0
    return {
        "retriever": getattr(retriever, "name", "unknown"),
        "n": n,
        "k": k,
        "tca_at_k": round(tca_k, 4),
        "mean_rank": round(mean_rank, 1),
        "mrr": round(mrr, 4),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_custom_retriever(path: str, corpus: list[dict]):
    """Load a custom retriever from an external Python file.

    The file must define a class named ``Retriever`` with:
      - ``__init__(self, corpus: list[dict])``
      - ``retrieve(self, query: str, corpus: list[dict], k: int) -> list[str]``
      - optional string attribute ``name``

    Example retriever file (my_retriever.py):
      class Retriever:
          name = "my_bm25_variant"
          def __init__(self, corpus):
              ...
          def retrieve(self, query, corpus, k):
              ...  # return list of doc_ids in ranked order
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_retriever", path)
    if spec is None:
        raise ImportError(f"Cannot load retriever from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Retriever"):
        raise AttributeError(
            f"{path} must define a class named 'Retriever' with an "
            "__init__(self, corpus) and retrieve(self, query, corpus, k) method."
        )
    return module.Retriever(corpus)


def main():
    parser = argparse.ArgumentParser(
        description="Score a retriever on FinSuperQA (TC-MQA benchmark).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 evaluate.py --retriever bm25
  python3 evaluate.py --retriever two_stage --split test --k 10
  python3 evaluate.py --retriever dense:intfloat/e5-large-v2 --hop_type type1
  python3 evaluate.py --retriever_path my_retriever.py --output results.json

Custom retriever (--retriever_path):
  Your file must define a class named Retriever with:
    def __init__(self, corpus: list[dict]): ...
    def retrieve(self, query: str, corpus: list[dict], k: int) -> list[str]: ...
  where the return value is a ranked list of doc_ids (best first).
""",
    )
    parser.add_argument("--retriever", default="bm25",
                        help="bm25 | two_stage | temporal_two_stage | dense:<model_id>  (ignored if --retriever_path set)")
    parser.add_argument("--retriever_path", default=None,
                        help="Path to a Python file defining a custom Retriever class")
    parser.add_argument("--split", default="all",
                        choices=["train", "dev", "test", "all"])
    parser.add_argument("--n_distractors", type=int, default=10,
                        help="Distractor events per example (default 10, matching paper corpus; 0 for no distractors)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", default=None,
                        help="Path for JSON output (default: stdout)")
    parser.add_argument("--hop_type", default=None,
                        help="Filter to hop type substring, e.g. 'type1'")
    args = parser.parse_args()

    print(f"Loading FinSuperQA ({args.split})...", file=sys.stderr)
    examples, corpus = load_finsuperqa(args.split, n_distractors=args.n_distractors)

    if args.hop_type:
        examples = [e for e in examples if args.hop_type in e.get("hop_type", "")]

    print(f"  {len(examples)} examples, {len(corpus)} corpus docs", file=sys.stderr)

    if args.retriever_path:
        print(f"Loading custom retriever from: {args.retriever_path}", file=sys.stderr)
        retriever = load_custom_retriever(args.retriever_path, corpus)
    else:
        retriever_id = args.retriever
        if retriever_id == "bm25":
            retriever = BM25Retriever(corpus)
        elif retriever_id in ("two_stage", "temporal_two_stage"):
            # two_stage is the published reference (TCA@5 = 0.978 on FinSuperQA all-1000).
            # temporal_two_stage is kept as an alias for backwards compatibility.
            retriever = TemporalTwoStageRetriever(corpus)
        elif retriever_id.startswith("dense:"):
            model_id = retriever_id[6:]
            retriever = DenseRetriever(model_id, corpus)
        else:
            print(f"Unknown retriever: {retriever_id}. "
                  f"Use bm25, two_stage, dense:<model_id>, or --retriever_path.",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Evaluating {getattr(retriever, 'name', args.retriever_path or args.retriever)}...",
          file=sys.stderr)
    result = evaluate(retriever, examples, corpus, k=args.k)

    out = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out)
        print(f"Saved → {args.output}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
