"""
FinSuperQA Retrieval Baselines
================================
Implements three baselines for the TC-MQA task:

1. BM25 — lexical relevance only, no temporal awareness
2. Dense (cosine sim on TF-IDF vectors) — semantic relevance only
3. Oracle — perfect retrieval of gold events (upper bound)

Plus our proposed system:
4. TemporalAuditRAG — dense retrieval + supersession graph + constrained re-ranking

All systems output a ranked list of event IDs.
The answer is derived from the top-k retrieved events using the supersession graph.
"""

from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from events import Event
from supersession_graph import SupersessionGraph, SupersessionEdge
from events import KnowledgeBase


# ---------------------------------------------------------------------------
# Utilities: simple TF-IDF (no external deps needed for demo)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


class TFIDF:
    """Minimal TF-IDF vectorizer."""

    def __init__(self, corpus: list[str]):
        self.corpus = corpus
        self.n_docs = len(corpus)
        self.token_docs = [Counter(tokenize(doc)) for doc in corpus]
        self.df: Counter = Counter()
        for td in self.token_docs:
            for tok in td:
                self.df[tok] += 1

    def tfidf(self, doc_idx: int) -> dict[str, float]:
        td = self.token_docs[doc_idx]
        total = sum(td.values())
        vec = {}
        for tok, cnt in td.items():
            tf = cnt / total
            idf = math.log((1 + self.n_docs) / (1 + self.df[tok])) + 1
            vec[tok] = tf * idf
        return vec

    def query_vector(self, query: str) -> dict[str, float]:
        toks = Counter(tokenize(query))
        total = sum(toks.values())
        vec = {}
        for tok, cnt in toks.items():
            tf = cnt / total
            idf = math.log((1 + self.n_docs) / (1 + self.df[tok])) + 1
            vec[tok] = tf * idf
        return vec

    def cosine(self, a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v ** 2 for v in a.values()))
        nb = math.sqrt(sum(v ** 2 for v in b.values()))
        return dot / (na * nb + 1e-12)


# ---------------------------------------------------------------------------
# BM25 (Robertson et al.)
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.doc_lens = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.doc_lens) / max(self.n, 1)
        self.df: Counter = Counter()
        for toks in self.tokenized:
            for tok in set(toks):
                self.df[tok] += 1

    def score(self, query: str, doc_idx: int) -> float:
        qtoks = tokenize(query)
        toks = self.tokenized[doc_idx]
        dl = self.doc_lens[doc_idx]
        tf_map = Counter(toks)
        score = 0.0
        for qt in qtoks:
            if qt not in tf_map:
                continue
            df = self.df.get(qt, 0)
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            tf = tf_map[qt]
            norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
            score += idf * norm
        return score

    def rank(self, query: str, event_ids: list[str]) -> list[str]:
        scores = [(eid, self.score(query, i)) for i, eid in enumerate(event_ids)]
        scores.sort(key=lambda x: -x[1])
        return [eid for eid, _ in scores]


# ---------------------------------------------------------------------------
# Dense retriever (TF-IDF cosine, stand-in for embedding similarity)
# ---------------------------------------------------------------------------

class DenseRetriever:
    def __init__(self, event_texts: list[str], event_ids: list[str]):
        self.event_ids = event_ids
        self.tfidf = TFIDF(event_texts)
        self.doc_vecs = [self.tfidf.tfidf(i) for i in range(len(event_texts))]

    def rank(self, query: str) -> list[str]:
        qvec = self.tfidf.query_vector(query)
        scores = [
            (eid, self.tfidf.cosine(qvec, dvec))
            for eid, dvec in zip(self.event_ids, self.doc_vecs)
        ]
        scores.sort(key=lambda x: -x[1])
        return [eid for eid, _ in scores]


# ---------------------------------------------------------------------------
# Oracle retriever
# ---------------------------------------------------------------------------

def oracle_rank(gold_event_ids: list[str], all_event_ids: list[str]) -> list[str]:
    """Perfect retrieval: gold events first, then rest in arbitrary order."""
    gold_set = set(gold_event_ids)
    ranked = list(gold_event_ids)  # gold first
    ranked += [eid for eid in all_event_ids if eid not in gold_set]
    return ranked


# ---------------------------------------------------------------------------
# TemporalAuditRAG
# ---------------------------------------------------------------------------

def temporal_audit_rerank(
    initial_ranking: list[str],
    supersession_graph: SupersessionGraph,
    event_id_map: dict[str, Event],
    k_candidates: int = 20,
) -> list[str]:
    """
    Given an initial ranking (from dense or BM25 retriever),
    apply supersession-aware re-ranking:

    For each event in the top-k_candidates:
      (a) If it is superseded, find its superseding event and add it to the pool
          (even if it would not have been retrieved by relevance alone)
      (b) Demote superseded events to the bottom of the ranking
      (c) Return the re-ranked list

    This is the core of TemporalAuditRAG's constrained retrieval.
    """
    candidate_ids = initial_ranking[:k_candidates]
    remaining_ids = initial_ranking[k_candidates:]

    # Build a lookup: event_id → event
    # Check which candidates are superseded and what supersedes them
    active = []
    superseded_with_trigger = []

    # Collect all triggered events we need to add
    added_trigger_ids = set()

    for eid in candidate_ids:
        event = event_id_map.get(eid)
        if event is None:
            active.append(eid)
            continue
        if supersession_graph.is_superseded(event):
            # Find the superseding event
            chain = supersession_graph.superseding_chain(event)
            if chain:
                trigger_id = chain[0].trigger.event_id
                superseded_with_trigger.append((eid, trigger_id))
                added_trigger_ids.add(trigger_id)
            else:
                superseded_with_trigger.append((eid, None))
        else:
            active.append(eid)

    # For each superseding trigger not yet in candidates, promote it from remaining
    promoted = []
    still_remaining = []
    for eid in remaining_ids:
        if eid in added_trigger_ids and eid not in candidate_ids:
            promoted.append(eid)
        else:
            still_remaining.append(eid)

    # Final ranking:
    # 1. Active (non-superseded) candidates in original order
    # 2. Promoted superseding events (newly surfaced from corpus)
    # 3. Superseded candidates (demoted but still returned for provenance)
    # 4. Remaining
    final = (
        active
        + promoted
        + [eid for eid, _ in superseded_with_trigger]
        + still_remaining
    )
    return final


# ---------------------------------------------------------------------------
# Answer derivation from retrieved events
# ---------------------------------------------------------------------------

def derive_answer(
    retrieved_ids: list[str],
    global_event_id_map: dict[str, Event],
    supersession_graph: SupersessionGraph,
    k: int = 5,
    query_employee: str | None = None,
    query_ticker: str | None = None,
) -> str:
    """
    Derive an answer from the top-k retrieved events using the supersession graph.

    Uses the FULL CORPUS event map — no case-local filtering by KB membership.

    Entity-scope filtering (query_employee, query_ticker): events whose
    (employee_id, ticker) do not match the query's extracted entities are treated
    as irrelevant context and excluded from the compliance decision. This is a
    legitimate query-processing step — the same entity extraction already used
    in the probe step — not oracle access. It prevents distractor events from
    other employees/securities from corrupting the answer.

    If extraction failed (query_employee or query_ticker is None), no entity
    filtering is applied (graceful degradation).

    Rule priority (highest first):
      1. Active EMERGENCY_EXCEPTION → compliant
      2. Active BLACKOUT_ANNOUNCED or WATCHLIST_ADDED → non_compliant
      3. Active PRE_CLEARANCE_APPROVED → active (valid clearance)
      4. Active BLACKOUT_LIFTED (with no valid pre-clearance) → non_compliant
      5. No relevant active events → requires_review
    """
    from rules import EventType, ComplianceStatus

    # Look up all top-k events from the full corpus map
    top_k = [global_event_id_map.get(eid) for eid in retrieved_ids[:k]]
    top_k = [e for e in top_k if e is not None]

    # Entity-scope filter: keep only events matching the query's (emp, ticker).
    # Consistent with the probe step — same extracted entities, same application.
    if query_employee is not None and query_ticker is not None:
        top_k = [
            e for e in top_k
            if e.employee_id == query_employee and e.security_ticker == query_ticker
        ]

    # An event is "active" if the per-example supersession graph does not mark it
    # as superseded. Non-case events are not in the graph → treated as active,
    # but are already removed by the entity filter above.
    active_events = [e for e in top_k if not supersession_graph.is_superseded(e)]

    has_exception = any(e.event_type == EventType.EMERGENCY_EXCEPTION for e in active_events)
    has_blackout  = any(e.event_type == EventType.BLACKOUT_ANNOUNCED for e in active_events)
    has_watchlist = any(e.event_type == EventType.WATCHLIST_ADDED for e in active_events)
    has_clearance = any(e.event_type == EventType.PRE_CLEARANCE_APPROVED for e in active_events)
    has_lift      = any(e.event_type == EventType.BLACKOUT_LIFTED for e in active_events)

    if has_exception:
        return ComplianceStatus.COMPLIANT.value
    if has_blackout or has_watchlist:
        return ComplianceStatus.NON_COMPLIANT.value
    if has_clearance:
        return ComplianceStatus.ACTIVE.value
    if has_lift and not has_clearance:
        return ComplianceStatus.NON_COMPLIANT.value

    return ComplianceStatus.REQUIRES_REVIEW.value
