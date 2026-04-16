"""
Contaminated-corpus ablation for Table~\ref{tab:contamination}.

Runs only the two systems that differ on a contaminated corpus:
  - temporal_audit_anchor_only  (expected T3 ≈ 0.624)
  - temporal_audit_plus / RSSG  (expected T3 ≈ 0.760)

Output: data/contaminated_corpus_results.json
"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "finsuperqa"))

from eval import load_examples, run_eval, aggregate_tca

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "finsuperqa_v1.jsonl")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "data", "contaminated_corpus_results.json")

examples = load_examples(DATA_PATH)
print(f"Dataset: {len(examples)} examples  |  enforce_entity_disjoint=False")

systems = ["temporal_audit_anchor_only", "temporal_audit_plus"]
all_results = {}

print(f"\n{'System':<28} {'R@5':>7} {'TCA':>7} {'T3':>7}")
print("-" * 55)

for system in systems:
    results = run_eval(examples, system=system, k=5, n_distractors=10,
                       enforce_entity_disjoint=False)
    agg = aggregate_tca(results)
    all_results[system] = agg

    ov  = agg["overall"]
    t3  = agg.get("type3_cross_session", {})
    print(
        f"{system:<28} "
        f"{ov['recall_at_k']:>7.4f} "
        f"{ov['tca']:>7.4f} "
        f"{t3.get('tca', float('nan')):>7.4f}"
    )

with open(OUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved → {OUT_PATH}")
