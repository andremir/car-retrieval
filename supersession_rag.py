"""
CAR: Generic Framework for Controlling Authority Retrieval
======================================================================

Separates the three concerns:
  1. Record model (timestamped item + free-form text + entity metadata)
  2. SupersessionRule (trigger_type supersedes target_type within a scope)
  3. DomainAdapter (pluggable entity extraction + rule registry)

The TwoStagePipeline works identically across domains:
  Stage 1: BM25 retrieves the "anchor" document (the one still using query vocabulary)
  Stage 2: Entity-indexed lookup finds the superseding document

Domain packs (thin adapters, ~5 lines each):
  SecurityAdapter  → scope = (product, cve_id)
  LegalAdapter     → scope = (case_slug,)
  FinanceAdapter   → scope = (employee_id, security_ticker)
"""
from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------

@dataclass
class Record:
    """
    A single timestamped document in a knowledge base.

    entities: arbitrary key→value pairs used for entity-indexed lookup.
    E.g.:
      security: {"product": "nginx", "cve_id": "CVE-2023-123"}
      legal:    {"case_slug": "roe_v_wade_410us113"}
      finance:  {"employee_id": "EMP042-Larsson", "security_ticker": "BAC"}
    """
    record_type: str
    timestamp: datetime
    text: str
    entities: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SupersessionRule:
    """
    A trigger_type event supersedes a target_type event within a shared scope.
    scope_keys: the entity keys that must match for the rule to apply.

    Example:
      SupersessionRule("patch_released", "cve_disclosed", ("product", "cve_id"))
      SupersessionRule("ruling_overruled", "ruling_issued", ("case_slug",))
      SupersessionRule("blackout_announced", "pre_clearance_approved", ("security_ticker",))
    """
    trigger_type: str
    target_type: str
    scope_keys: tuple[str, ...]


# ---------------------------------------------------------------------------
# Utility: BM25 tokenizer + scorer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class _BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.df = Counter(t for doc in corpus for t in set(doc))
        self.idf = {t: math.log((self.n - df + 0.5) / (df + 0.5) + 1)
                    for t, df in self.df.items()}
        self.corpus = corpus

    def scores(self, query: str) -> list[float]:
        qtoks = _tokenize(query)
        result = []
        for doc in self.corpus:
            dl = len(doc)
            tf = Counter(doc)
            s = sum(
                self.idf.get(t, 0) * tf.get(t, 0) * (self.k1 + 1) /
                (tf.get(t, 0) + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                for t in set(qtoks)
            )
            result.append(s)
        return result


# ---------------------------------------------------------------------------
# Entity index
# ---------------------------------------------------------------------------

class EntityIndex:
    """Maps a tuple of entity values → list of record indices."""

    def __init__(self, records: list[Record], scope_keys: tuple[str, ...]):
        self._index: dict[tuple, list[int]] = {}
        for i, rec in enumerate(records):
            key = tuple(
                str(rec.entities.get(k, "")).lower().strip()
                for k in scope_keys
            )
            if all(key):  # skip records missing any scope key
                self._index.setdefault(key, []).append(i)

    def lookup(self, entities: dict[str, Any], scope_keys: tuple[str, ...]) -> list[int]:
        key = tuple(
            str(entities.get(k, "")).lower().strip()
            for k in scope_keys
        )
        return self._index.get(key, [])


# ---------------------------------------------------------------------------
# Two-stage pipeline
# ---------------------------------------------------------------------------

class TwoStagePipeline:
    """
    Stage 1: BM25 retrieves the top-k1 documents from the corpus.
    Stage 2: For each Stage-1 result whose record_type matches anchor_type,
             use its entity metadata to look up superseding records
             (those with superseding_type).
    Final ranking: superseding candidates (in Stage-1 order) + remaining docs.
    """

    def __init__(
        self,
        corpus: list[Record],
        anchor_type: str,         # type of the "anchor" document (Stage-1 target)
        superseding_type: str,    # type of the gold superseding document (Stage-2 target)
        scope_keys: tuple[str, ...],
    ):
        self.corpus = corpus
        self.anchor_type = anchor_type
        self.superseding_type = superseding_type
        self.scope_keys = scope_keys

        texts = [r.text for r in corpus]
        self._bm25 = _BM25([_tokenize(t) for t in texts])
        self._entity_idx = EntityIndex(corpus, scope_keys)

    def retrieve(self, query: str, k1: int = 5) -> list[int]:
        """Return final ranking of record indices."""
        raw_scores = self._bm25.scores(query)
        stage1_ranking = sorted(range(len(self.corpus)), key=lambda i: -raw_scores[i])

        promoted, seen = [], set()
        for doc_idx in stage1_ranking[:k1]:
            rec = self.corpus[doc_idx]
            if rec.record_type == self.anchor_type:
                candidates = self._entity_idx.lookup(rec.entities, self.scope_keys)
                for cand_idx in candidates:
                    if (self.corpus[cand_idx].record_type == self.superseding_type
                            and cand_idx not in seen):
                        promoted.append(cand_idx)
                        seen.add(cand_idx)

        remaining = [i for i in stage1_ranking if i not in seen]
        return promoted + remaining

    def tca_at_k(self, query: str, gold_idx: int, k: int = 5, k1: int = 5) -> int:
        ranking = self.retrieve(query, k1=k1)
        return int(gold_idx in ranking and ranking.index(gold_idx) < k)

    def bm25_rank(self, query: str) -> list[int]:
        raw_scores = self._bm25.scores(query)
        return sorted(range(len(self.corpus)), key=lambda i: -raw_scores[i])


# ---------------------------------------------------------------------------
# Domain adapters
# ---------------------------------------------------------------------------

class DomainAdapter:
    """Base class — override for each domain."""
    anchor_type: str      = ""   # e.g. "cve_disclosed"
    superseding_type: str = ""   # e.g. "patch_released"
    scope_keys: tuple     = ()   # e.g. ("product", "cve_id")

    def extract_query_entities(self, query: str) -> dict[str, str]:
        """Extract entity values from a structured or free-form query."""
        return {}


class SecurityAdapter(DomainAdapter):
    """
    Security domain: CVE disclosure → patch release note.
    Scope: (product, cve_id).
    Structured queries contain both; free-form queries may have neither
    (use two-stage via BM25 to find disclosure first).
    """
    anchor_type      = "cve_disclosed"
    superseding_type = "patch_released"
    scope_keys       = ("product", "cve_id")

    _CVE_RE  = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
    _PROD_RE = re.compile(r"Is (.+?) still affected", re.IGNORECASE)

    def extract_query_entities(self, query: str) -> dict[str, str]:
        cve_m  = self._CVE_RE.search(query)
        prod_m = self._PROD_RE.search(query)
        return {
            "cve_id":  cve_m.group(0).upper()  if cve_m  else "",
            "product": prod_m.group(1).strip() if prod_m else "",
        }


class LegalAdapter(DomainAdapter):
    """
    Legal domain: original ruling → overruling opinion.
    Scope: (case_slug,).
    Free-form queries describe the legal topic; Stage-1 BM25 finds the original ruling.
    """
    anchor_type      = "ruling_issued"
    superseding_type = "ruling_overruled"
    scope_keys       = ("case_slug",)

    _CITE_RE = re.compile(
        r"([A-Z][a-z]+(?:\s+v\.\s+[A-Z][a-z]+)?),\s+\d+ [A-Z]\.\S+ \d+",
        re.IGNORECASE
    )

    def extract_query_entities(self, query: str) -> dict[str, str]:
        m = self._CITE_RE.search(query)
        if m:
            slug = re.sub(r"[^a-z0-9]+", "_", m.group(0).lower())[:60]
            return {"case_slug": slug}
        return {}


class FinanceAdapter(DomainAdapter):
    """
    Finance compliance domain: pre-clearance approval → blackout announcement.
    Scope: (employee_id, security_ticker).
    """
    anchor_type      = "pre_clearance_approved"
    superseding_type = "blackout_announced"
    scope_keys       = ("employee_id", "security_ticker")

    _EMP_RE    = re.compile(r"EMP\d{3}-\w+", re.IGNORECASE)
    _TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")

    def extract_query_entities(self, query: str) -> dict[str, str]:
        emp_m = self._EMP_RE.search(query)
        ticker_m = self._TICKER_RE.search(query)
        return {
            "employee_id":      emp_m.group(0).upper()    if emp_m    else "",
            "security_ticker":  ticker_m.group(1).upper() if ticker_m else "",
        }


# ---------------------------------------------------------------------------
# Convenience: build TwoStagePipeline from adapter
# ---------------------------------------------------------------------------

def make_pipeline(corpus: list[Record], adapter: DomainAdapter) -> TwoStagePipeline:
    return TwoStagePipeline(
        corpus=corpus,
        anchor_type=adapter.anchor_type,
        superseding_type=adapter.superseding_type,
        scope_keys=adapter.scope_keys,
    )
