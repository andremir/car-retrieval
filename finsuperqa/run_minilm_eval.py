"""
Run FinSuperQA evaluation with MiniLM sentence-transformer dense retrieval.
Produces the same per-hop-type TCA breakdown as eval.py ("bm25"/"dense" systems)
but uses real MiniLM embeddings instead of TF-IDF cosine.

Output: data/eval_results_minilm.json
Also patches: data/eval_results_v1.json  (adds "dense_minilm" key)

Usage:
  cd finsuperqa && python3 run_minilm_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from eval import build_eval_corpus, build_per_example_sg, extract_entities_from_query
from baselines import oracle_rank, derive_answer
from tca import compute_tca, aggregate_tca

DATA_PATH = ROOT / "data" / "finsuperqa_v1.jsonl"
OUT_PATH  = ROOT / "data" / "eval_results_minilm.json"
V1_PATH   = ROOT / "data" / "eval_results_v1.json"

MODEL_ID   = "sentence-transformers/all-MiniLM-L6-v2"
K          = 5
N_DIST     = 10
BATCH_SIZE = 256


def load_examples() -> list[dict]:
    return [json.loads(l) for l in open(DATA_PATH) if l.strip()]


def run_minilm_eval(examples: list[dict]) -> dict:
    print(f"Building eval corpus (n_distractors={N_DIST}, entity_disjoint=True)...")
    all_events, event_id_map, gold_map = build_eval_corpus(
        examples, n_distractors=N_DIST, enforce_entity_disjoint=True,
    )
    corpus_texts = [e.raw_text for e in all_events]
    corpus_ids   = [e.event_id for e in all_events]
    print(f"  Corpus: {len(corpus_texts)} events")

    print(f"Loading {MODEL_ID} and encoding corpus...")
    model = SentenceTransformer(MODEL_ID)
    doc_embs = model.encode(
        corpus_texts, batch_size=BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=True,
    )

    results = []
    queries = [ex["query"] for ex in examples]
    print("Encoding queries...")
    q_embs = model.encode(
        queries, batch_size=BATCH_SIZE,
        normalize_embeddings=True, show_progress_bar=False,
    )

    print("Ranking and computing TCA...")
    for ex, q_emb in zip(examples, q_embs):
        query = ex["query"]
        gold_ids = gold_map[ex["example_id"]]

        # Cosine ranking (embeddings are L2-normalised → dot product = cosine)
        scores  = doc_embs @ q_emb
        ranking = [corpus_ids[i] for i in np.argsort(-scores)]

        # Per-example supersession graph (for TCA + answer derivation)
        ex_sg = build_per_example_sg(ex)
        ex_sg_edges = [
            {"trigger_id": edge.trigger.event_id, "target_id": edge.target.event_id}
            for edge in ex_sg.edges
        ]
        ex_eid_map = {e.event_id: e for e in ex_sg.kb.events}

        q_emp, q_ticker = extract_entities_from_query(query)
        predicted = derive_answer(
            ranking, event_id_map, ex_sg, k=K,
            query_employee=q_emp, query_ticker=q_ticker,
        )
        result = compute_tca(
            example=ex,
            retrieved_event_ids=ranking,
            predicted_answer=predicted,
            supersession_graph_edges=ex_sg_edges,
            k=K,
        )
        results.append(result)

    return aggregate_tca(results)


def main():
    examples = load_examples()
    print(f"Loaded {len(examples)} examples")

    agg = run_minilm_eval(examples)
    print("\n=== MiniLM Dense Results ===")
    print(f"Overall TCA:  {agg['overall']['tca']:.4f}")
    print(f"T0: {agg['type0_no_supersession']['tca']:.4f}")
    print(f"T1: {agg['type1_simple_supersession']['tca']:.4f}")
    print(f"T2: {agg['type2_chain_supersession']['tca']:.4f}")
    print(f"T3: {agg['type3_cross_session']['tca']:.4f}")

    # Save standalone results
    with open(OUT_PATH, "w") as f:
        json.dump({"model": MODEL_ID, "k": K, "n_distractors": N_DIST,
                   "results": agg}, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")

    # Patch eval_results_v1.json
    v1 = json.load(open(V1_PATH))
    v1["dense_minilm"] = agg
    with open(V1_PATH, "w") as f:
        json.dump(v1, f, indent=2)
    print(f"Patched → {V1_PATH}  (added 'dense_minilm' key)")


if __name__ == "__main__":
    main()
