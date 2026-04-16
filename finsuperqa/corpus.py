"""
FinSuperQA Corpus Builder
==========================
Wraps each isolated scenario with a shared distractor corpus.

Without distractors, the retrieval task is trivial (only 2-3 events in KB).
With distractors:
  - Other employees' pre-clearances for the same security
  - Same employee's events for other securities
  - Events that look temporally relevant but aren't
  - Older pre-clearances that were superseded in DIFFERENT scenarios

A standard RAG system must retrieve the RIGHT events (matching employee+ticker+time)
and also handle the supersession signal — which it cannot by relevance alone.

Corpus is shared: each query sees the FULL corpus (all examples' events).
This mimics a real compliance KB with thousands of conversations.
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta

from rules import EventType
from events import Event, KnowledgeBase
from generator import (
    EMPLOYEES, SECURITIES, COUNTERPARTIES, POLICIES, render_nl, _base_time
)


def build_shared_corpus(
    scenarios: list[dict],
    n_distractors_per_scenario: int = 10,
    rng: random.Random = None,
    enforce_entity_disjoint: bool = True,
) -> tuple[list[Event], dict[str, list[str]]]:
    """
    Build a shared event corpus from all scenarios + distractors.

    Args:
        enforce_entity_disjoint: If True (default), distractors are filtered so
            that no distractor shares an (employee, ticker) pair with any gold
            example. This eliminates T3 contamination and allows anchor-based
            probing to achieve TCA=1.000. Set to False to reproduce the
            contaminated-corpus ablation (Table~\ref{tab:contamination}) where
            entity-overlapping distractors reduce T3 TCA to 0.624 (anchor-only)
            or 0.760 (anchor+RSSG). The contaminated mode mirrors production
            corpora where entity pair reuse is unavoidable.

    Returns:
        corpus_events: all events (scenario events + distractors), sorted by time
        example_gold_ids: {example_id → [event_ids that are gold for this query]}
    """
    if rng is None:
        rng = random.Random(99)

    all_events: list[Event] = []
    example_gold_ids: dict[str, list[str]] = {}

    # 1. Add all scenario events and record gold event ids
    for ex in scenarios:
        gold_ids = []
        for kb_event in ex["kb"]:
            e = Event(
                event_type=EventType(kb_event["event_type"]),
                timestamp=datetime.fromisoformat(kb_event["timestamp"]),
                session_id=ex.get("example_id", "unknown"),
                turn_id=0,
                employee_id=ex["employee"],
                security_ticker=ex.get("ticker"),
                raw_text=kb_event["text"],
                event_id=kb_event["event_id"],
            )
            all_events.append(e)
            gold_ids.append(kb_event["event_id"])
        example_gold_ids[ex["example_id"]] = gold_ids

    # Build the set of all gold (emp, ticker) pairs — distractors must not share
    # these pairs when enforce_entity_disjoint=True, otherwise they contaminate
    # entity-scoped RSSG queries with fake events that masquerade as real
    # compliance history for those entities.
    gold_pairs: set[tuple] = {(ex["employee"], ex.get("ticker")) for ex in scenarios}

    # 2. Add distractors for each scenario
    for ex in scenarios:
        emp = ex["employee"]
        ticker = ex.get("ticker")
        t0 = datetime.fromisoformat(ex["kb"][0]["timestamp"])
        sess = "distract-" + str(uuid.uuid4())[:6]

        for d_idx in range(n_distractors_per_scenario):
            # Vary: same employee/different ticker, different employee/same ticker,
            # or completely unrelated event.
            choice = d_idx % 3
            if choice == 0:
                # Same employee, different security.
                if enforce_entity_disjoint:
                    # Filter: ensure (emp, other_ticker) is not any gold pair — a
                    # mode-0 distractor for example (EMP_A, TICKER_Y) could pick
                    # TICKER_X if that is a gold ticker for EMP_A in another example,
                    # contaminating RSSG.
                    safe_tickers = [s for s in SECURITIES
                                    if s != ticker and (emp, s) not in gold_pairs]
                    if not safe_tickers:
                        safe_tickers = [s for s in SECURITIES if s != ticker]
                else:
                    safe_tickers = [s for s in SECURITIES if s != ticker]
                d_ticker = rng.choice(safe_tickers)
                d_emp = emp
            elif choice == 1:
                # Different employee, same security.
                if enforce_entity_disjoint:
                    # Filter: exclude employees whose (other_emp, ticker) is a gold pair.
                    safe_emps = [e for e in EMPLOYEES
                                 if e != emp and (e, ticker) not in gold_pairs]
                    if not safe_emps:
                        safe_emps = [e for e in EMPLOYEES if e != emp]
                else:
                    safe_emps = [e for e in EMPLOYEES if e != emp]
                d_emp = rng.choice(safe_emps)
                d_ticker = ticker
            else:
                # Completely unrelated.
                if enforce_entity_disjoint:
                    # Retry until we get a non-gold pair.
                    for _ in range(200):
                        d_emp = rng.choice(EMPLOYEES)
                        d_ticker = rng.choice(SECURITIES)
                        if (d_emp, d_ticker) not in gold_pairs:
                            break
                else:
                    d_emp = rng.choice(EMPLOYEES)
                    d_ticker = rng.choice(SECURITIES)

            # Random event type from the pool (no supersession events for distractors
            # to avoid accidentally creating false ground truth)
            d_type = rng.choice([
                EventType.PRE_CLEARANCE_APPROVED,
                EventType.PRE_CLEARANCE_DENIED,
                EventType.POLICY_ACKNOWLEDGED,
                EventType.CONFLICT_DISCLOSED,
            ])
            d_time = t0 + timedelta(days=rng.randint(-5, 5))
            d_event = Event(
                event_type=d_type,
                timestamp=d_time,
                session_id=sess,
                turn_id=d_idx,
                employee_id=d_emp,
                security_ticker=d_ticker,
                raw_text=render_nl(d_type, d_emp, d_ticker, None, None, d_time),
            )
            all_events.append(d_event)

    # Sort full corpus by time
    all_events.sort(key=lambda e: e.timestamp)

    return all_events, example_gold_ids


def evaluate_retrieval(
    retrieved_event_ids: list[str],
    gold_event_ids: list[str],
    k: int = 5,
) -> dict:
    """Standard retrieval metrics at rank k."""
    retrieved_k = retrieved_event_ids[:k]
    hits = sum(1 for eid in retrieved_k if eid in set(gold_event_ids))
    recall_at_k = hits / max(len(gold_event_ids), 1)
    precision_at_k = hits / max(k, 1)
    return {
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "hits": hits,
        "gold_n": len(gold_event_ids),
        "retrieved_k": k,
    }
