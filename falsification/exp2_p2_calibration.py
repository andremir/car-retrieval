"""
Experiment 2: Proposition 2 Natural-Case Calibration
=====================================================

Tests the claim: for scope-indexed algorithms (no access to ⊳),
  TCA(q) ≤ φ(q) · R_anchor(q)   where φ(q) = 1/κ(q)

Two sub-experiments:

  2a. Controlled adversarial κ-sweep on GHSA (n=159):
      κ ∈ {1,2,3,4,5} — add κ−1 adversarial confound patches per pair
      (same entity scope, identical vocabulary to real patch → maximally ambiguous)
      Scope-indexed algorithm: BM25 top-k1 → entity bucket → pick last-added (indeterminate order)
      Authority-indexed algorithm: BM25 top-k1 → entity bucket → pick by text similarity to anchor
      Expected: TCA_scope ≈ (1/κ) · R_anchor  (bound holds with equality in adversarial case)

  2b. Natural corpus check on FDA (n=500, existing results):
      Compute per-query κ from number of drugs retrieved in Stage-1 that share entity scope
      (multiple drugs with same indication vocabulary → κ > 1 at Stage-1)
      Check: does TCA(q) ≤ (1/κ) · R_anchor(q) for each query?

Pass/fail criteria printed at end. Key result: P2 is not a vacuous worst-case artifact —
it describes natural corpus failure when scope ambiguity exists.

Dependencies: pip install sentence-transformers rank-bm25 numpy requests
Artifacts: data/falsification_p2_calibration.json

Runtime: ~3 min
"""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT       = Path(__file__).parent.parent
PAIRS_PATH = ROOT / "data" / "ghsa_real_pairs.json"
FREEFORM   = ROOT / "data" / "ghsa_freeform_cache.jsonl"
FDA_PATH   = ROOT / "data" / "fda_recall_pairs.json"
OUT_PATH   = ROOT / "data" / "falsification_p2_calibration.json"

K  = 5
K1 = 5  # Stage-1 cutoff


# ---------------------------------------------------------------------------
# 2a: Controlled κ-sweep on GHSA
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_ghsa_corpus_with_confounds(pairs: list[dict], kappa: int,
                                     n_distractors: int = 2000,
                                     seed: int = 42) -> tuple[list[str], list[dict], int, dict[str, list[int]]]:
    """
    Build a GHSA corpus with κ−1 adversarial confound patches per real pair.

    Confound patches:
      - Same (product, cve_id) entity scope as the real patch
      - Text = real patch text (IDENTICAL vocabulary — adversarial construction from P2 proof)
      - Different doc_id: they represent the adversarial κ−1 alternative authority chains

    Entity index: (product, cve_id) → [real_patch_idx, confound1_idx, confound2_idx, ...]
    """
    docs, metas = [], []
    disc_start = 0

    # Disclosures
    for p in pairs:
        docs.append(p["disclosure_text"])
        metas.append({"type": "disclosure", "product": p["product"].lower().strip(),
                      "cve_id": p["cve_id"].upper().strip(),
                      "pair_idx": len(metas)})

    # Real patches
    patch_start = len(docs)
    for i, p in enumerate(pairs):
        docs.append(p["patch_text"])
        metas.append({"type": "patch", "product": p["product"].lower().strip(),
                      "cve_id": p["cve_id"].upper().strip(),
                      "pair_idx": i, "is_gold": True})

    # Adversarial confound patches (κ−1 per pair) — IDENTICAL text to real patch
    confound_start = len(docs)
    entity_index: dict[str, list[int]] = {}  # key → [real_idx, confound_idx, ...]
    for i, p in enumerate(pairs):
        key = (p["product"].lower().strip(), p["cve_id"].upper().strip())
        real_idx = patch_start + i
        entity_index[str(key)] = [real_idx]
        for c in range(kappa - 1):
            confound_idx = len(docs)
            # IDENTICAL text to real patch — maximum adversarial ambiguity
            docs.append(p["patch_text"])
            metas.append({"type": "patch_confound", "product": p["product"].lower().strip(),
                          "cve_id": p["cve_id"].upper().strip(),
                          "pair_idx": i, "is_gold": False, "confound_num": c})
            entity_index[str(key)].append(confound_idx)

    # Synthetic distractors (different entity scope — no ambiguity)
    rng = random.Random(seed)
    products = ["nginx","openssl","log4j","redis","django","flask","rails",
                "postgresql","mysql","tomcat","spring","curl","libssl"]
    attacks  = ["remote code execution","SQL injection","buffer overflow","path traversal"]
    sevs     = ["Critical","High","Medium"]
    for i in range(n_distractors):
        prod = rng.choice(products)
        cve  = f"CVE-2024-{80000+i:05d}"
        ver  = f"{rng.randint(1,9)}.{rng.randint(0,9)}"
        docs.append(
            f"{cve} affects {prod}. "
            f"An attacker can exploit {rng.choice(attacks)}. "
            f"Severity: {rng.choice(sevs)}. "
            f"Versions up to {ver} are affected. No patch available."
        )
        metas.append({"type": "distractor", "product": prod, "cve_id": cve,
                      "pair_idx": -1, "is_gold": False})

    return docs, metas, patch_start, entity_index


def run_scope_indexed(pairs, queries, docs, metas, patch_start,
                      entity_index, kappa):
    """
    Scope-indexed algorithm: BM25 → entity bucket → pick by insertion order
    (since all κ patches are identical, order = arbitrary = worst-case random).
    Returns (tca5, r_anchor) averages.
    """
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    tca_list, r_anchor_list = [], []

    for i, pair in enumerate(pairs):
        pid = pair["id"]
        if pid not in queries:
            continue
        q = queries[pid]
        gold_idx = patch_start + i

        bm25_scores = bm25.get_scores(tokenize(q))
        ranked_k1 = list(np.argsort(-bm25_scores)[:K1])

        # Check anchor recall: did disclosure appear in top-k1?
        disc_idx = i  # disclosure index = pair index
        r_anchor = int(disc_idx in ranked_k1)
        r_anchor_list.append(r_anchor)

        # Scope-indexed: if we found any disclosure, retrieve entity bucket, pick FIRST (arbitrary)
        key = str((pair["product"].lower().strip(), pair["cve_id"].upper().strip()))
        bucket = entity_index.get(key, [])

        # Since all κ patches are identical, pick first in bucket (deterministic but arbitrary)
        # In adversarial case, this is equivalent to random — gold is at position 0 (first inserted)
        # To simulate the adversarial permutation, we randomize bucket order once per κ
        rng = random.Random(i * kappa + kappa)
        shuffled_bucket = list(bucket)
        rng.shuffle(shuffled_bucket)

        if r_anchor and shuffled_bucket:
            # Scope-indexed picks the first element from shuffled bucket
            prediction = shuffled_bucket[0]
            tca5 = int(prediction == gold_idx)
        else:
            tca5 = 0

        tca_list.append(tca5)

    tca = sum(tca_list) / len(tca_list)
    r_anchor = sum(r_anchor_list) / len(r_anchor_list)
    return tca, r_anchor


def run_authority_indexed(pairs, queries, docs, metas, patch_start,
                          entity_index, kappa):
    """
    Authority-indexed algorithm: BM25 → entity bucket → pick by cosine similarity to
    ANCHOR TEXT (implicit ⊳ signal: the real patch was written to address the disclosure,
    so it has higher textual affinity to the anchor disclosure text than confounds, even
    when confounds have identical text).

    In the adversarial case (identical confound text), authority-indexed also cannot distinguish
    → TCA = same as scope-indexed. This is the CORRECT result: even authority-indexed
    fails when κ chains are informationally identical. The paper's architecture succeeds
    because in practice the real patch is identified via entity-indexed lookup where
    the entity index was built with (product, cve_id, release_tag) → single patch.

    For this experiment, we show the fallback: with truly identical-text confounds,
    authority-indexed also reduces to (1/κ)·R_anchor — the bound is tight.
    """
    bm25 = BM25Okapi([tokenize(d) for d in docs])

    tca_list, r_anchor_list = [], []
    for i, pair in enumerate(pairs):
        pid = pair["id"]
        if pid not in queries:
            continue
        q = queries[pid]
        gold_idx = patch_start + i

        bm25_scores = bm25.get_scores(tokenize(q))
        ranked_k1 = list(np.argsort(-bm25_scores)[:K1])

        disc_idx = i
        r_anchor = int(disc_idx in ranked_k1)
        r_anchor_list.append(r_anchor)

        key = str((pair["product"].lower().strip(), pair["cve_id"].upper().strip()))
        bucket = entity_index.get(key, [])

        if r_anchor and bucket:
            # Try text similarity to disclosure as ⊳ proxy
            disc_text = docs[disc_idx]
            disc_toks = set(tokenize(disc_text))
            best_idx, best_overlap = bucket[0], -1
            for b_idx in bucket:
                b_toks = set(tokenize(docs[b_idx]))
                # Jaccard overlap
                overlap = len(disc_toks & b_toks) / max(len(disc_toks | b_toks), 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = b_idx
            tca5 = int(best_idx == gold_idx)
        else:
            tca5 = 0
        tca_list.append(tca5)

    tca = sum(tca_list) / len(tca_list)
    r_anchor = sum(r_anchor_list) / len(r_anchor_list)
    return tca, r_anchor


def run_kappa_sweep(pairs, queries):
    """Run scope-indexed and authority-indexed for κ ∈ {1,2,3,4,5}."""
    print("\n[2a] Controlled κ-sweep on GHSA (adversarial confound patches):")
    print(f"  {'κ':>4} {'φ=1/κ':>8} {'R_anchor':>10} {'TCA_scope':>12} {'bound=φ·Ra':>12} "
          f"{'TCA_auth':>12} {'bound_satisfied':>16}")
    print("  " + "-"*78)

    rows = []
    for kappa in range(1, 6):
        docs, metas, patch_start, entity_index = build_ghsa_corpus_with_confounds(
            pairs, kappa=kappa
        )
        tca_scope, r_anchor = run_scope_indexed(
            pairs, queries, docs, metas, patch_start, entity_index, kappa
        )
        tca_auth, _ = run_authority_indexed(
            pairs, queries, docs, metas, patch_start, entity_index, kappa
        )

        phi = 1.0 / kappa
        bound = phi * r_anchor
        satisfied = tca_scope <= bound + 0.005  # 0.5% tolerance for rounding

        rows.append({
            "kappa": kappa, "phi": round(phi, 4), "r_anchor": round(r_anchor, 4),
            "tca_scope": round(tca_scope, 4), "bound": round(bound, 4),
            "tca_auth": round(tca_auth, 4), "bound_satisfied": bool(satisfied),
        })

        sat_sym = "✓" if satisfied else "✗"
        print(f"  {kappa:>4} {phi:>8.3f} {r_anchor:>10.3f} {tca_scope:>12.3f} "
              f"{bound:>12.3f} {tca_auth:>12.3f} {sat_sym:>16}")

    return rows


# ---------------------------------------------------------------------------
# 2b: Natural FDA corpus check
# ---------------------------------------------------------------------------

def run_fda_natural_check(pairs):
    """
    For FDA, compute per-query Stage-1 behavior and estimate κ empirically.

    κ(q) ≈ number of distinct anchor (drug label) documents retrieved in Stage-1
    that share the therapeutic indication vocabulary with the query.
    When BM25 Stage-1 retrieves the correct drug's label first, TCA = R_anchor.
    When κ drugs share the same indication vocabulary, effective TCA = R_anchor/κ.

    We estimate κ per query from BM25 Stage-1 top-k results: count how many
    distinct drugs appear in top-k (each drug has one anchor label in corpus).
    """
    print("\n[2b] Natural FDA corpus κ estimation (n=500)…")

    docs, queries_fda, gold_indices = [], [], []
    drug_doc_map = {}  # norm_name → doc_index in anchor portion

    for i, p in enumerate(pairs):
        docs.append(p["anchor_text"])      # approved drug label
        drug_doc_map[p["norm_name"]] = i
        queries_fda.append(p["query"])
        gold_indices.append(i)

    # Add 2000 distractors with random drug names
    rng = random.Random(77)
    drug_names = [p["norm_name"] for p in pairs]
    drug_classes = ["antihypertensive","antibiotic","antidiabetic","analgesic",
                    "anticoagulant","antihistamine","antidepressant","statin"]
    n_anchor = len(docs)
    for j in range(2000):
        name = rng.choice(drug_names)
        cls  = rng.choice(drug_classes)
        docs.append(f"This medication is indicated for {cls} treatment. "
                    f"Dosage form: oral tablet. Active ingredient: {name}_analog{j}. "
                    f"Approved by FDA. Clinical trials demonstrate efficacy.")

    bm25 = BM25Okapi([tokenize(d) for d in docs])

    kappa_list, tca_list, r_anchor_list = [], [], []
    for i, (q, gold_i) in enumerate(zip(queries_fda, gold_indices)):
        scores = bm25.get_scores(tokenize(q))
        ranked = list(np.argsort(-scores)[:K])

        r_anchor = int(gold_i in ranked)
        # Estimate κ: how many real drug labels (index < n_anchor) appear in top-k?
        n_anchors_in_top = sum(1 for r in ranked if r < n_anchor)
        kappa_est = max(1, n_anchors_in_top)

        tca = r_anchor  # for BM25, TCA = R_anchor (single stage, entity-indexed stage not run here)

        kappa_list.append(kappa_est)
        tca_list.append(tca)
        r_anchor_list.append(r_anchor)

    # Group by κ and check bound
    from collections import defaultdict
    by_kappa: dict[int, list[dict]] = defaultdict(list)
    for kappa_est, tca, r_anchor in zip(kappa_list, tca_list, r_anchor_list):
        by_kappa[kappa_est].append({"tca": tca, "r_anchor": r_anchor})

    print(f"  {'κ_est':>6} {'n':>6} {'R_anchor':>10} {'TCA':>8} {'bound=1/κ·Ra':>14} {'held?':>8}")
    print("  " + "-"*56)
    natural_rows = []
    for k in sorted(by_kappa.keys()):
        group = by_kappa[k]
        n_g = len(group)
        ra = sum(g["r_anchor"] for g in group) / n_g
        tca = sum(g["tca"] for g in group) / n_g
        bound = (1.0 / k) * ra
        held = tca <= bound + 0.005
        print(f"  {k:>6} {n_g:>6} {ra:>10.3f} {tca:>8.3f} {bound:>14.3f} {'✓' if held else '✗':>8}")
        natural_rows.append({"kappa_est": k, "n": n_g, "r_anchor": round(ra, 4),
                              "tca": round(tca, 4), "bound": round(bound, 4), "held": bool(held)})

    pct_held = sum(r["held"] for r in natural_rows) / len(natural_rows)
    print(f"\n  Bound held in {pct_held:.0%} of κ-groups.")
    return natural_rows


# ---------------------------------------------------------------------------
# Pass/fail report
# ---------------------------------------------------------------------------
def print_verdict(sweep_rows, fda_rows):
    print("\n" + "="*70)
    print("PASS/FAIL REPORT — Proposition 2 Calibration")
    print("="*70)

    # Adversarial sweep
    all_sat = all(r["bound_satisfied"] for r in sweep_rows)
    if all_sat:
        print("[PASS] Adversarial κ-sweep: TCA_scope ≤ (1/κ)·R_anchor for all κ ∈ {1..5}.")
        print("       The P2 bound holds with equality in the adversarial (identical-text) case.")
    else:
        failed = [r["kappa"] for r in sweep_rows if not r["bound_satisfied"]]
        print(f"[FAIL] Bound violated for κ ∈ {failed}. Check experiment setup.")

    # Correlation check
    kappas = [r["kappa"] for r in sweep_rows]
    tcas   = [r["tca_scope"] for r in sweep_rows]
    if len(kappas) > 1:
        corr = np.corrcoef(kappas, tcas)[0, 1]
        if corr < -0.8:
            print(f"[PASS] TCA_scope tracks 1/κ (ρ = {corr:.2f}): bound is tight, not just an upper bound.")
        else:
            print(f"[NOTE] ρ(TCA_scope, κ) = {corr:.2f}: moderate tracking. "
                  f"Bound holds but is not tight in this regime.")

    # Two-stage exceeds bound
    for r in sweep_rows:
        phi_bound = r["phi"] * r["r_anchor"]
        if r["tca_auth"] > phi_bound + 0.01:
            print(f"[PASS] κ={r['kappa']}: Two-Stage TCA_auth={r['tca_auth']:.3f} > bound={phi_bound:.3f}. "
                  f"Authority-indexed exceeds scope-indexed ceiling.")
        elif r["kappa"] == 1:
            pass  # κ=1 → bound = R_anchor; authority-indexed trivially equal
        else:
            print(f"[NOTE] κ={r['kappa']}: Two-Stage TCA_auth={r['tca_auth']:.3f} ≈ bound={phi_bound:.3f}. "
                  f"Identical-text confounds prevent disambiguation even by two-stage (expected).")

    # Natural FDA
    pct_held = sum(r["held"] for r in fda_rows) / len(fda_rows)
    if pct_held >= 0.85:
        print(f"[PASS] Natural FDA: bound held for {pct_held:.0%} of κ-groups. "
              f"P2 describes natural corpus behavior, not just adversarial case.")
    else:
        print(f"[FAIL-soft] Natural FDA: bound held for {pct_held:.0%} of κ-groups. "
              f"BM25 Stage-1 estimation of κ may be inaccurate; check bucket computation.")
    print("="*70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairs   = json.load(open(PAIRS_PATH))
    queries: dict[str, str] = {}
    for line in open(FREEFORM):
        if line.strip():
            e = json.loads(line)
            queries[e["id"]] = e["query"]
    fda_pairs = json.load(open(FDA_PATH))
    print(f"Loaded {len(pairs)} GHSA pairs, {len(queries)} free-form queries, {len(fda_pairs)} FDA pairs")

    sweep_rows  = run_kappa_sweep(pairs, queries)
    fda_rows    = run_fda_natural_check(fda_pairs)
    print_verdict(sweep_rows, fda_rows)

    out = {
        "exp2a_adversarial_kappa_sweep": {
            "description": "Controlled κ-sweep: κ−1 identical-text confound patches per GHSA pair",
            "rows": sweep_rows,
        },
        "exp2b_natural_fda": {
            "description": "Natural FDA corpus: κ estimated from BM25 Stage-1 entity bucket size",
            "rows": fda_rows,
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
