"""
Temporal Compliance Accuracy (TCA) Metric
==========================================
TCA extends standard Recall@k with a supersession validity check.

Standard Recall@k:  did we retrieve the gold events?
TCA:                did we retrieve the gold events AND flag any superseded
                    evidence we retrieved (i.e., we did NOT ignore superseding events)?

Formally (Definition 3 in paper):
  TCA(system, KB, query) = 1
    iff (a) the returned answer is correct
    AND (b) for every event e in the retrieved set,
            if e is superseded in KB then the superseding event e' is also retrieved

A system that retrieves ONLY the (now-void) pre-clearance scores:
  - Recall@k = 1  (it found the "relevant" document)
  - TCA = 0       (it ignored the blackout that voids the pre-clearance)

This is the core failure mode that motivates the paper.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TCAResult:
    example_id: str
    hop_type: str

    # Standard retrieval
    recall_at_k: float
    precision_at_k: float

    # Supersession compliance
    retrieved_superseded_ids: list[str]     # voided events that were retrieved
    retrieved_superseding_ids: list[str]    # the events that void them (also retrieved?)
    missing_superseding_ids: list[str]      # superseding events NOT retrieved → TCA failure

    # TCA
    answer_correct: bool
    no_ignored_supersession: bool           # True if all superseding events are retrieved
    tca: float                              # 1.0 or 0.0 (binary per example)

    # Provenance completeness
    provenance_recall: float                # fraction of gold provenance hops retrieved


def compute_tca(
    example: dict,
    retrieved_event_ids: list[str],
    predicted_answer: str,
    supersession_graph_edges: list[dict],   # [{trigger_id, target_id}]
    k: int = 5,
) -> TCAResult:
    """
    Compute TCA for one example.

    Args:
        example: a FinSuperQA example dict
        retrieved_event_ids: ordered list of retrieved event IDs (by rank)
        predicted_answer: the system's predicted answer string
        supersession_graph_edges: list of {trigger_id, target_id} dicts
                                  (from the supersession graph over the full corpus)
        k: cutoff for retrieval
    """
    retrieved_k = set(retrieved_event_ids[:k])
    gold_ids = set(e["event_id"] for e in example["kb"])
    gold_k_hits = retrieved_k & gold_ids

    recall_at_k = len(gold_k_hits) / max(len(gold_ids), 1)
    precision_at_k = len(gold_k_hits) / max(k, 1)

    # Build supersession lookup: target_id → trigger_id
    supersedes: dict[str, str] = {
        edge["target_id"]: edge["trigger_id"]
        for edge in supersession_graph_edges
    }

    # For each retrieved event, check if it is superseded
    retrieved_superseded = []
    missing_superseding = []
    retrieved_superseding_ids = []

    for eid in retrieved_k:
        if eid in supersedes:
            # This event is superseded by supersedes[eid]
            retrieved_superseded.append(eid)
            trigger_id = supersedes[eid]
            if trigger_id in retrieved_k:
                retrieved_superseding_ids.append(trigger_id)
            else:
                # The superseding event was NOT retrieved → TCA failure
                missing_superseding.append(trigger_id)

    no_ignored_supersession = len(missing_superseding) == 0
    answer_correct = predicted_answer.strip().lower() == example["answer"].strip().lower()
    tca = 1.0 if (answer_correct and no_ignored_supersession) else 0.0

    # Provenance recall: how many hops of the gold provenance chain were retrieved?
    gold_provenance_ids = set(h["event_id"] for h in example["provenance"])
    provenance_hits = retrieved_k & gold_provenance_ids
    provenance_recall = len(provenance_hits) / max(len(gold_provenance_ids), 1)

    return TCAResult(
        example_id=example["example_id"],
        hop_type=example["hop_type"],
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        retrieved_superseded_ids=retrieved_superseded,
        retrieved_superseding_ids=retrieved_superseding_ids,
        missing_superseding_ids=missing_superseding,
        answer_correct=answer_correct,
        no_ignored_supersession=no_ignored_supersession,
        tca=tca,
        provenance_recall=provenance_recall,
    )


def aggregate_tca(results: list[TCAResult]) -> dict:
    """Aggregate TCA results across examples, broken down by hop type."""
    from collections import defaultdict

    by_type: dict[str, list[TCAResult]] = defaultdict(list)
    for r in results:
        by_type[r.hop_type].append(r)

    def summarize(rs: list[TCAResult]) -> dict:
        n = len(rs)
        return {
            "n": n,
            "recall_at_k": sum(r.recall_at_k for r in rs) / n,
            "tca": sum(r.tca for r in rs) / n,
            "answer_accuracy": sum(r.answer_correct for r in rs) / n,
            "supersession_compliance": sum(r.no_ignored_supersession for r in rs) / n,
            "provenance_recall": sum(r.provenance_recall for r in rs) / n,
        }

    out = {"overall": summarize(results)}
    for hop_type, rs in sorted(by_type.items()):
        out[hop_type] = summarize(rs)

    return out
