"""
Cross-Domain TC-MQA Evaluation
================================
Runs all 5 retrieval systems across three domains:
  1. Financial Compliance (FinSuperQA)
  2. CVE/Security-Patch  (CVEPatchQA)
  3. Legal Precedent     (LegalPrecedentQA)

For CVE and Legal domains: evaluated on T0 + T1 (250 examples each).
FinSuperQA results loaded from eval_results_v1.json (already computed).

Key claim to validate:
  BM25 and Dense achieve TCA = 0.000 on T1 across ALL three domains.
  TemporalAuditRAG++ (anchor + RSSG) recovers to high TCA on T1 in all three.

Output: data/cross_domain_results.json
"""

from __future__ import annotations
import json
import re
import sys
import random
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# repo root on path for package imports (cvepatchqa, legalprecedentqa)
# finsuperqa/ on path for shared utilities (BM25, DenseRetriever, etc.)
_repo_root = str(Path(__file__).parent)
_finsuperqa_dir = str(Path(__file__).parent / "finsuperqa")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
if _finsuperqa_dir not in sys.path:
    sys.path.insert(0, _finsuperqa_dir)
from baselines import BM25, DenseRetriever, oracle_rank
from supersession_graph import SupersessionGraph
from events import Event, KnowledgeBase
from tca import compute_tca, aggregate_tca, TCAResult


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def build_corpus_from_examples(
    examples: list[dict],
    domain_event_type_cls,
    n_distractors: int,
    distractor_fn,
    seed: int = 99,
) -> tuple[list[Event], dict[str, Event], dict[str, list[str]]]:
    """Build the retrieval corpus for a domain."""
    rng = random.Random(seed)
    gold_map: dict[str, list[str]] = {}
    all_events: list[Event] = []
    seen_ids: set[str] = set()

    for ex in examples:
        gold_ids = []
        for kb_ev in ex["kb"]:
            eid = kb_ev["event_id"]
            if eid not in seen_ids:
                e = Event(
                    event_type=domain_event_type_cls(kb_ev["event_type"]),
                    timestamp=datetime.fromisoformat(kb_ev["timestamp"]),
                    session_id=ex["example_id"],
                    turn_id=0,
                    employee_id=ex["employee"],
                    security_ticker=ex.get("ticker"),
                    raw_text=kb_ev["text"],
                    event_id=eid,
                )
                all_events.append(e)
                seen_ids.add(eid)
            gold_ids.append(eid)
        gold_map[ex["example_id"]] = gold_ids

    # Add distractors
    distractors = distractor_fn(examples, n_distractors=n_distractors, rng=rng)
    for d in distractors:
        e = Event(
            event_type=domain_event_type_cls(d["event_type"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            session_id=d.get("session_id", "distract"),
            turn_id=0,
            employee_id=d["employee"],
            security_ticker=d.get("ticker"),
            raw_text=d["text"],
            event_id=d["event_id"],
        )
        all_events.append(e)

    all_events.sort(key=lambda e: e.timestamp)
    event_id_map = {e.event_id: e for e in all_events}

    n_gold = sum(len(gids) for gids in gold_map.values())
    n_distract = len(all_events) - n_gold
    print(f"  Corpus: {n_gold} gold + {n_distract} distractors = {len(all_events)} total")

    return all_events, event_id_map, gold_map


def build_per_example_sg(
    ex: dict,
    domain_event_type_cls,
    domain_supersession_rules: list,
) -> SupersessionGraph:
    """Build isolated supersession graph for one example (for TCA + answer derivation)."""
    kb = KnowledgeBase()
    for kb_ev in ex["kb"]:
        e = Event(
            event_type=domain_event_type_cls(kb_ev["event_type"]),
            timestamp=datetime.fromisoformat(kb_ev["timestamp"]),
            session_id=ex["example_id"],
            turn_id=0,
            employee_id=ex["employee"],
            security_ticker=ex.get("ticker"),
            raw_text=kb_ev["text"],
            event_id=kb_ev["event_id"],
        )
        kb.add(e)
    sg = SupersessionGraph(kb=kb)
    sg.build(rules=domain_supersession_rules)
    return sg


def derive_answer_generic(
    retrieved_ids: list[str],
    global_event_id_map: dict[str, Event],
    answer_sg: SupersessionGraph,
    k: int,
    query_employee: str | None,
    query_ticker: str | None,
    gold_answer: str,
) -> str:
    """
    Domain-generic answer derivation.
    Filters top-k to matching entity scope, finds active events, returns answer
    based on the example's gold answer derivation pattern.
    """
    top_k = [global_event_id_map.get(eid) for eid in retrieved_ids[:k]]
    top_k = [e for e in top_k if e is not None]

    # Entity-scope filter
    if query_employee:
        top_k = [e for e in top_k if e.employee_id == query_employee]
    if query_ticker:
        top_k = [e for e in top_k if e.security_ticker == query_ticker]

    active = [e for e in top_k if not answer_sg.is_superseded(e)]

    if not active:
        # No active events in scope → cannot determine
        # Return a wrong answer (the predicted answer that TCA will penalize)
        return "__no_active_events__"

    # Answer is determined by the "strongest" active event type
    # For CVE domain: PATCH_RELEASED/PATCH_SUPERSEDED → "patched"; CVE_DISCLOSED → "vulnerable"
    # For Legal domain: RULING_OVERRULED → "overruled"; RULING_ISSUED → "good_law"
    # We determine the answer state from the active event types present
    active_types = {e.event_type.value for e in active}

    # CVE domain
    if "patch_released" in active_types or "patch_superseded" in active_types:
        return "patched"
    if "workaround_issued" in active_types:
        return "mitigated"
    if "cve_reopened" in active_types:
        return "vulnerable"
    if "cve_disclosed" in active_types:
        return "vulnerable"

    # Legal domain
    if "ruling_overruled" in active_types:
        return "overruled"
    if "ruling_codified" in active_types:
        return "codified"
    if "ruling_issued" in active_types or "ruling_affirmed" in active_types:
        return "good_law"

    # Fallback
    return list(active_types)[0]


def temporal_audit_plus_full_corpus(
    initial_ranking: list[str],
    ex_sg: SupersessionGraph,
    global_event_id_map: dict[str, Event],
    same_entity_events: list[Event],
    k_candidates: int = 20,
) -> list[str]:
    """Anchor-based probe (same code as finsuperqa/eval.py)."""
    candidate_ids = set(initial_ranking[:k_candidates])
    promoted_ids: list[str] = []
    for eid in initial_ranking[:k_candidates]:
        event = global_event_id_map.get(eid)
        if event is None:
            continue
        for candidate_e in same_entity_events:
            if (candidate_e.event_id not in candidate_ids
                    and candidate_e.timestamp > event.timestamp):
                promoted_ids.append(candidate_e.event_id)
                candidate_ids.add(candidate_e.event_id)

    kb_id_to_event = {e.event_id: e for e in ex_sg.kb.events}
    promoted_set = set(promoted_ids)
    tier1, tier2, tier3, tier4 = [], [], [], []
    for eid in promoted_ids:
        kb_e = kb_id_to_event.get(eid)
        if kb_e and not ex_sg.is_superseded(kb_e):
            tier1.append(eid)
    for eid in initial_ranking[:k_candidates]:
        if eid in promoted_set:
            continue
        kb_e = kb_id_to_event.get(eid)
        if kb_e is None:
            tier3.append(eid)
        elif ex_sg.is_superseded(kb_e):
            tier4.append(eid)
        else:
            tier2.append(eid)
    return tier1 + tier2 + tier3 + tier4 + initial_ranking[k_candidates:]


def run_domain_eval(
    examples: list[dict],
    domain_event_type_cls,
    domain_supersession_rules: list,
    distractor_fn,
    system: str,
    k: int = 5,
    n_distractors: int = 10,
) -> list[TCAResult]:
    """Run one system over one domain."""
    all_events, event_id_map, gold_map = build_corpus_from_examples(
        examples, domain_event_type_cls, n_distractors, distractor_fn
    )

    corpus_texts = [e.raw_text for e in all_events]
    corpus_ids = [e.event_id for e in all_events]

    # Entity index for anchor probe
    entity_index: dict[tuple, list[Event]] = defaultdict(list)
    for e in all_events:
        entity_index[(e.employee_id, e.security_ticker)].append(e)

    if system == "bm25":
        _bm25 = BM25(corpus_texts)
        get_ranking = lambda query, gold_ids: _bm25.rank(query, corpus_ids)
    elif system == "dense":
        _dense = DenseRetriever(corpus_texts, corpus_ids)
        get_ranking = lambda query, gold_ids: _dense.rank(query)
    elif system == "oracle":
        get_ranking = lambda query, gold_ids: oracle_rank(gold_ids, corpus_ids)
    else:
        _dense = DenseRetriever(corpus_texts, corpus_ids)
        get_ranking = None

    results = []
    for ex in examples:
        query = ex["query"]
        gold_ids = gold_map[ex["example_id"]]

        # Per-example supersession graph (uses domain-specific rules via build(rules=...))
        ex_sg = build_per_example_sg(ex, domain_event_type_cls, domain_supersession_rules)
        ex_sg_edges = [
            {"trigger_id": edge.trigger.event_id, "target_id": edge.target.event_id}
            for edge in ex_sg.edges
        ]
        answer_sg = ex_sg

        if system in ("temporal_audit_anchor_only", "temporal_audit_plus"):
            initial = _dense.rank(query)
            same_entity = entity_index.get((ex["employee"], ex.get("ticker")), [])
            ranked = temporal_audit_plus_full_corpus(
                initial_ranking=initial,
                ex_sg=ex_sg,
                global_event_id_map=event_id_map,
                same_entity_events=same_entity,
                k_candidates=20,
            )
            if system == "temporal_audit_plus" and same_entity:
                # RSSG: build local supersession graph from entity-scoped events
                local_kb = KnowledgeBase()
                for e in same_entity:
                    local_kb.add(e)
                local_sg = SupersessionGraph(kb=local_kb)
                local_sg.build(rules=domain_supersession_rules)
                answer_sg = local_sg
        else:
            ranked = get_ranking(query, gold_ids)

        predicted = derive_answer_generic(
            ranked, event_id_map, answer_sg, k,
            query_employee=ex["employee"],
            query_ticker=ex.get("ticker"),
            gold_answer=ex["answer"],
        )

        # compute_tca expects "provenance" field (full chain for ProvRec metric)
        # For T0/T1 cross-domain examples, provenance = all KB events
        ex_with_prov = dict(ex)
        if "provenance" not in ex_with_prov:
            ex_with_prov["provenance"] = ex["kb"]

        result = compute_tca(
            example=ex_with_prov,
            retrieved_event_ids=ranked,
            predicted_answer=predicted,
            supersession_graph_edges=ex_sg_edges,
            k=k,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    k = 5
    n_distractors = 10

    # ---- Load domain modules (repo root is on sys.path) ----
    import cvepatchqa.generator as cve_gen
    import cvepatchqa.rules as cve_rules
    import legalprecedentqa.generator as leg_gen
    import legalprecedentqa.rules as leg_rules

    # ---- Generate datasets ----
    print("=== CVE/Security-Patch Domain ===")
    cve_examples = cve_gen.generate_dataset(n_t0=250, n_t1=250)
    cve_data_path = Path(__file__).parent / "data" / "cvepatchqa_v1.jsonl"
    with open(cve_data_path, "w") as f:
        for ex in cve_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Generated {len(cve_examples)} examples")

    print("\n=== Legal Precedent Domain ===")
    leg_examples = leg_gen.generate_dataset(n_t0=250, n_t1=250)
    leg_data_path = Path(__file__).parent / "data" / "legalprecedentqa_v1.jsonl"
    with open(leg_data_path, "w") as f:
        for ex in leg_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Generated {len(leg_examples)} examples")

    systems = ["bm25", "dense", "oracle", "temporal_audit_anchor_only", "temporal_audit_plus"]
    system_labels = {
        "bm25": "BM25",
        "dense": "Dense",
        "oracle": "Oracle",
        "temporal_audit_anchor_only": "TAudit+ (anchor)",
        "temporal_audit_plus": "TAudit++ (RSSG)",
    }
    hop_types = ["type0_no_supersession", "type1_patch_supersedes_disclosure",
                 "type1_overruling_supersedes_ruling"]

    all_results = {}

    for domain_name, examples, event_type_cls, sup_rules, distract_fn in [
        ("cve_security_patch", cve_examples, cve_rules.EventType,
         cve_rules.SUPERSESSION_RULES, cve_gen.build_distractor_events),
        ("legal_precedent", leg_examples, leg_rules.EventType,
         leg_rules.SUPERSESSION_RULES, leg_gen.build_distractor_events),
    ]:
        print(f"\n{'='*60}")
        print(f"Domain: {domain_name}")
        print(f"{'='*60}")
        domain_results = {}

        print(f"\n{'System':<28} {'R@5':>7} {'TCA':>7} {'Acc':>7}")
        print("-" * 52)

        for system in systems:
            results = run_domain_eval(
                examples, event_type_cls, sup_rules, distract_fn,
                system=system, k=k, n_distractors=n_distractors,
            )
            agg = aggregate_tca(results)
            domain_results[system] = agg

            ov = agg["overall"]
            label = system_labels.get(system, system)
            print(f"{label:<28} {ov['recall_at_k']:>7.4f} {ov['tca']:>7.4f} {ov['answer_accuracy']:>7.4f}")

        print("\nPer-type TCA:")
        for ht in ["type0_no_supersession",
                   "type1_patch_supersedes_disclosure",
                   "type1_overruling_supersedes_ruling"]:
            if ht in domain_results.get("bm25", {}):
                print(f"  {ht}:")
                for system in systems:
                    v = domain_results.get(system, {}).get(ht, {}).get("tca", float("nan"))
                    label = system_labels.get(system, system)
                    print(f"    {label:<28} {v:.4f}")

        all_results[domain_name] = domain_results

    # ---- Load FinSuperQA results ----
    finsuperqa_path = Path(__file__).parent / "data" / "eval_results_v1.json"
    with open(finsuperqa_path) as f:
        finsuperqa_results = json.load(f)
    # Rename keys to match cross-domain output format
    fin_mapped = {}
    key_map = {
        "bm25": "bm25",
        "dense": "dense",
        "oracle": "oracle",
        "temporal_audit_anchor_only": "temporal_audit_anchor_only",
        "temporal_audit_plus": "temporal_audit_plus",
    }
    for sys_key, mapped_key in key_map.items():
        if sys_key in finsuperqa_results:
            fin_mapped[mapped_key] = finsuperqa_results[sys_key]
    all_results["financial_compliance"] = fin_mapped

    # ---- Save results ----
    out_path = Path(__file__).parent / "data" / "cross_domain_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved cross-domain results → {out_path}")

    # ---- Print cross-domain summary table ----
    print("\n\n=== CROSS-DOMAIN SUMMARY (T1 TCA) ===")
    print(f"{'System':<28}", end="")
    domain_labels = {
        "financial_compliance": "Finance T1",
        "cve_security_patch": "CVE T1",
        "legal_precedent": "Legal T1",
    }
    t1_keys = {
        "financial_compliance": "type1_simple_supersession",
        "cve_security_patch": "type1_patch_supersedes_disclosure",
        "legal_precedent": "type1_overruling_supersedes_ruling",
    }
    for d in domain_labels:
        print(f"  {domain_labels[d]:>12}", end="")
    print()
    print("-" * 66)
    for system in systems:
        label = system_labels.get(system, system)
        print(f"{label:<28}", end="")
        for domain, t1_key in t1_keys.items():
            v = all_results.get(domain, {}).get(system, {}).get(t1_key, {}).get("tca", float("nan"))
            print(f"  {v:>12.4f}", end="")
        print()


if __name__ == "__main__":
    main()
