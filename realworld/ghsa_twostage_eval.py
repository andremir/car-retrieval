"""
Two-stage pipeline evaluation on 159 GHSA pairs with FREE-FORM queries.

Experiments:
  Exp 2: Baselines (BM25, MiniLM, E5-large) on free-form queries
  Exp 3: Two-stage pipeline (BM25 or Dense → entity-indexed patch lookup)
  Exp 4: Stage-1 ablation (BM25 vs Dense for disclosure retrieval)
  Exp 5: HyDE on free-form queries (reads existing hyde cache if available)
  Exp 6: Error analysis of Stage-1 failures

Two-stage pipeline:
  Stage 1: Retrieve top-k1 documents using query against full 2318-doc corpus
  Stage 2: For each Stage-1 result that is a disclosure, use its CVE ID →
           entity-indexed lookup → find the paired patch note → return as candidate
  Final ranking: Stage-2 patch candidates first (in Stage-1 order), then rest

Output: data/ghsa_twostage_results.json
"""
from __future__ import annotations
import json, re, math, time, random, os
from pathlib import Path
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
PAIRS_PATH  = ROOT / "data" / "ghsa_real_pairs.json"
FREEFORM    = ROOT / "data" / "ghsa_freeform_cache.jsonl"
HYDE_CACHE  = ROOT / "data" / "ghsa_hyde_cache.jsonl"
OUT_PATH    = ROOT / "data" / "ghsa_twostage_results.json"

# ---------------------------------------------------------------------------
# Corpus construction (identical to ghsa_retrieval_eval.py)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_corpus(pairs: list[dict], n_distractors: int = 2000):
    docs, doc_metas = [], []

    # Real disclosures
    disc_start = 0
    for p in pairs:
        docs.append(p["disclosure_text"])
        doc_metas.append({"type": "disclosure", "product": p["product"],
                          "cve_id": p["cve_id"], "pair_idx": len(doc_metas)})

    # Real patch notes
    patch_start = len(docs)
    for i, p in enumerate(pairs):
        docs.append(p["patch_text"])
        doc_metas.append({"type": "patch", "product": p["product"],
                          "cve_id": p["cve_id"], "pair_idx": i})

    # Synthetic distractors (reproducible)
    rng = random.Random(42)
    products = ["nginx","openssl","log4j","redis","django","flask","rails",
                "postgresql","mysql","tomcat","spring","hibernate","curl","wget",
                "libssl","libjpeg","zlib","expat","glibc","libxml2"]
    severities = ["Critical","High","Medium"]
    attacks = ["remote code execution","SQL injection","buffer overflow",
               "path traversal","cross-site scripting","privilege escalation"]
    for i in range(n_distractors):
        product = rng.choice(products)
        severity = rng.choice(severities)
        attack = rng.choice(attacks)
        cve = f"CVE-2023-{90000+i:05d}"
        ver = f"{rng.randint(1,9)}.{rng.randint(0,9)}"
        text = (
            f"{cve} has been identified in {product}. "
            f"The flaw permits {attack} by unauthenticated remote attackers. "
            f"Severity: {severity}. "
            f"Affected versions: {product} up to and including {ver}. "
            f"No patch is currently available. Exploitation has been observed in the wild. "
            f"This vulnerability remains unmitigated at time of publication."
        )
        docs.append(text)
        doc_metas.append({"type": "distractor", "product": product, "cve_id": cve,
                          "pair_idx": -1})

    return docs, doc_metas, disc_start, patch_start


# ---------------------------------------------------------------------------
# Entity index: (product, cve_id) → patch_doc_idx
# ---------------------------------------------------------------------------
def build_entity_index(doc_metas: list[dict], patch_start: int) -> dict[tuple, int]:
    idx = {}
    for i in range(patch_start, len(doc_metas)):
        m = doc_metas[i]
        if m["type"] == "patch":
            key = (m["product"].lower().strip(), m["cve_id"].upper().strip())
            idx[key] = i
    return idx


# ---------------------------------------------------------------------------
# Two-stage retrieval
# ---------------------------------------------------------------------------
def twostage_retrieve(
    query: str,
    stage1_ranking: list[int],
    doc_metas: list[dict],
    entity_index: dict[tuple, int],
    k1: int,
) -> list[int]:
    """
    For the top-k1 Stage-1 results:
      - if it's a disclosure → look up its patch note via entity index
      - collect those patch_idxs in Stage-1 order
    Then return [patch candidates from entity lookup] + [remaining Stage-1 docs]
    """
    stage1_top = stage1_ranking[:k1]
    patch_candidates = []
    seen = set()

    for doc_idx in stage1_top:
        meta = doc_metas[doc_idx]
        if meta["type"] == "disclosure":
            key = (meta["product"].lower().strip(), meta["cve_id"].upper().strip())
            patch_idx = entity_index.get(key)
            if patch_idx is not None and patch_idx not in seen:
                patch_candidates.append(patch_idx)
                seen.add(patch_idx)

    # Append remaining (non-duplicate, not already in patch_candidates)
    remaining = [d for d in stage1_ranking if d not in seen]
    return patch_candidates + remaining


# ---------------------------------------------------------------------------
# TCA@k
# ---------------------------------------------------------------------------
def tca_at_k(ranking: list[int], gold_idx: int, k: int) -> int:
    return int(ranking.index(gold_idx) + 1 <= k) if gold_idx in ranking else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairs = json.load(open(PAIRS_PATH))
    print(f"Loaded {len(pairs)} pairs")

    # Load free-form queries
    freeform: dict[str, str] = {}
    for line in FREEFORM.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            freeform[e["id"]] = e["query"]
    print(f"Loaded {len(freeform)} free-form queries")

    # Load HyDE hypotheticals (for Exp 5)
    hyde_cache: dict[str, str] = {}
    if HYDE_CACHE.exists():
        for line in HYDE_CACHE.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                hyde_cache[e["id"]] = e["hypothetical"]
        print(f"Loaded {len(hyde_cache)} HyDE hypotheticals")

    # Build corpus
    docs, doc_metas, disc_start, patch_start = build_corpus(pairs)
    print(f"Corpus: {len(docs)} docs (disc={patch_start}, patch_start={patch_start}, total={len(docs)})")

    # Entity index
    entity_index = build_entity_index(doc_metas, patch_start)
    print(f"Entity index: {len(entity_index)} (product, cve_id) → patch pairs")

    # Build BM25 on full corpus
    print("Building BM25 index...")
    tokenized = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)

    # Build dense models
    print("Loading MiniLM...")
    minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Encoding corpus with MiniLM...")
    minilm_embs = minilm.encode(docs, normalize_embeddings=True,
                                show_progress_bar=False, batch_size=64)
    print(f"MiniLM embeddings: {minilm_embs.shape}")

    print("Loading E5-large...")
    e5 = SentenceTransformer("intfloat/e5-large-v2")
    print("Encoding corpus with E5-large (this takes a few minutes)...")
    e5_embs = e5.encode(
        [f"passage: {d}" for d in docs],
        normalize_embeddings=True, show_progress_bar=False, batch_size=32
    )
    print(f"E5-large embeddings: {e5_embs.shape}")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    results_per_pair = []
    k = 5
    k1_values = [5, 10, 20]

    for i, pair in enumerate(pairs):
        pid = pair["id"]
        gold_idx = patch_start + i  # gold = patch note for this pair

        ff_query = freeform.get(pid)
        if not ff_query:
            print(f"  [{i+1:>3}] SKIP: no free-form query")
            continue

        # -------------------------------------------------------------------
        # Stage-1 rankings for free-form query
        # -------------------------------------------------------------------
        # BM25 stage-1
        bm25_scores = bm25.get_scores(tokenize(ff_query))
        bm25_ranking = list(np.argsort(-bm25_scores))

        # MiniLM stage-1
        q_emb_minilm = minilm.encode([ff_query], normalize_embeddings=True)
        minilm_scores = (q_emb_minilm @ minilm_embs.T)[0]
        minilm_ranking = list(np.argsort(-minilm_scores))

        # E5-large stage-1
        q_emb_e5 = e5.encode([f"query: {ff_query}"], normalize_embeddings=True)
        e5_scores = (q_emb_e5 @ e5_embs.T)[0]
        e5_ranking = list(np.argsort(-e5_scores))

        # HyDE stage-1 (if available)
        hyp = hyde_cache.get(pid)
        if hyp:
            hyp_emb = minilm.encode([hyp], normalize_embeddings=True)
            hyde_scores = (hyp_emb @ minilm_embs.T)[0]
            hyde_ranking = list(np.argsort(-hyde_scores))
        else:
            hyde_ranking = None

        # -------------------------------------------------------------------
        # Exp 2: Baseline TCA@k on free-form
        # -------------------------------------------------------------------
        bm25_ff_rank = bm25_ranking.index(gold_idx) + 1
        minilm_ff_rank = minilm_ranking.index(gold_idx) + 1
        e5_ff_rank = e5_ranking.index(gold_idx) + 1

        # -------------------------------------------------------------------
        # Exp 3: Two-stage TCA@k for various k1
        # -------------------------------------------------------------------
        twostage_results = {}
        for k1 in k1_values:
            for s1_name, s1_rank in [("bm25", bm25_ranking), ("minilm", minilm_ranking)]:
                ts_ranking = twostage_retrieve(ff_query, s1_rank, doc_metas, entity_index, k1)
                ts_rank = ts_ranking.index(gold_idx) + 1
                twostage_results[f"{s1_name}_k{k1}"] = {"rank": ts_rank, "tca5": int(ts_rank <= k)}

        # -------------------------------------------------------------------
        # Stage-1 disclosure recall (what fraction of stage-1 top-k1 are the
        # correct disclosure for this pair?)
        # -------------------------------------------------------------------
        disc_idx = disc_start + i  # correct disclosure doc
        disc_recall = {}
        for k1 in k1_values:
            for s1_name, s1_rank in [("bm25", bm25_ranking), ("minilm", minilm_ranking)]:
                disc_recall[f"{s1_name}_k{k1}"] = int(disc_idx in s1_rank[:k1])

        # -------------------------------------------------------------------
        # Exp 4: Stage-1 ablation — BM25 vs Dense for disclosure retrieval
        # -------------------------------------------------------------------
        bm25_disc_rank  = bm25_ranking.index(disc_idx) + 1
        minilm_disc_rank = minilm_ranking.index(disc_idx) + 1
        e5_disc_rank     = e5_ranking.index(disc_idx) + 1

        # -------------------------------------------------------------------
        # Exp 5: HyDE on free-form
        # -------------------------------------------------------------------
        hyde_ff_rank = None
        if hyde_ranking is not None:
            hyde_ff_rank = hyde_ranking.index(gold_idx) + 1

        # -------------------------------------------------------------------
        # Exp 6: Error analysis for k1=10, BM25 two-stage
        # -------------------------------------------------------------------
        ts_bm25_k10 = twostage_results["bm25_k10"]
        ts_failed = ts_bm25_k10["tca5"] == 0
        if ts_failed:
            # Was disc retrieved in top-10?
            disc_in_top10 = disc_recall["bm25_k10"]
            if not disc_in_top10:
                error_cat = "stage1_miss"   # BM25 didn't find disclosure in top-10
            else:
                error_cat = "stage2_miss"   # Disc found but patch mapping failed (shouldn't happen)
        else:
            error_cat = "success"

        results_per_pair.append({
            "id": pid,
            "cve_id": pair["cve_id"],
            "product": pair["product"],
            "freeform_query": ff_query,
            # Exp 2: baselines
            "bm25_ff_rank": bm25_ff_rank, "bm25_ff_tca5": int(bm25_ff_rank <= k),
            "minilm_ff_rank": minilm_ff_rank, "minilm_ff_tca5": int(minilm_ff_rank <= k),
            "e5_ff_rank": e5_ff_rank, "e5_ff_tca5": int(e5_ff_rank <= k),
            # Exp 3: two-stage
            "twostage": twostage_results,
            # Exp 4: stage-1 disc retrieval
            "bm25_disc_rank": bm25_disc_rank,
            "minilm_disc_rank": minilm_disc_rank,
            "e5_disc_rank": e5_disc_rank,
            "disc_recall": disc_recall,
            # Exp 5: HyDE
            "hyde_ff_rank": hyde_ff_rank,
            "hyde_ff_tca5": int(hyde_ff_rank <= k) if hyde_ff_rank else None,
            # Exp 6: error
            "error_cat": error_cat,
        })

        if (i + 1) % 20 == 0 or i < 3:
            ts_k10 = twostage_results["bm25_k10"]["rank"]
            print(f"  [{i+1:>3}/{len(pairs)}] bm25_ff={bm25_ff_rank:>5} "
                  f"dense={minilm_ff_rank:>5} ts_k10={ts_k10:>4} err={error_cat}")

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    n = len(results_per_pair)
    print(f"\n=== Results (n={n}, k={k}) ===")

    def agg(key):
        vals = [r[key] for r in results_per_pair if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    print(f"\n--- Exp 2: Baselines on free-form queries ---")
    bm25_tca  = agg("bm25_ff_tca5")
    min_tca   = agg("minilm_ff_tca5")
    e5_tca    = agg("e5_ff_tca5")
    print(f"BM25    TCA@5: {bm25_tca:.4f}")
    print(f"MiniLM  TCA@5: {min_tca:.4f}")
    print(f"E5-large TCA@5: {e5_tca:.4f}")

    print(f"\n--- Exp 3: Two-stage pipeline (BM25 Stage-1) ---")
    for k1 in k1_values:
        key = f"bm25_k{k1}"
        tca = sum(r["twostage"][key]["tca5"] for r in results_per_pair) / n
        mean_rank = sum(r["twostage"][key]["rank"] for r in results_per_pair) / n
        disc_rec = sum(r["disc_recall"][key] for r in results_per_pair) / n
        print(f"  k1={k1:>2}: TCA@5={tca:.4f}  mean_rank={mean_rank:.1f}  disc_recall={disc_rec:.4f}")

    print(f"\n--- Exp 4: Stage-1 ablation (disclosure retrieval) ---")
    for s1_name in ["bm25", "minilm"]:
        rank_key = f"{s1_name}_disc_rank"
        ranks = [r[rank_key] for r in results_per_pair]
        disc_recall_1 = sum(1 for rk in ranks if rk <= 1) / n
        disc_recall_5 = sum(1 for rk in ranks if rk <= 5) / n
        disc_recall_10 = sum(1 for rk in ranks if rk <= 10) / n
        mean_rk = sum(ranks) / n
        print(f"  {s1_name.upper():8} disc_recall@1={disc_recall_1:.4f}  @5={disc_recall_5:.4f}  @10={disc_recall_10:.4f}  mean_rank={mean_rk:.1f}")

    print(f"\n--- Exp 5: HyDE on free-form queries ---")
    hyde_avail = [r for r in results_per_pair if r["hyde_ff_tca5"] is not None]
    if hyde_avail:
        hyde_tca = sum(r["hyde_ff_tca5"] for r in hyde_avail) / len(hyde_avail)
        print(f"HyDE TCA@5: {hyde_tca:.4f} (n={len(hyde_avail)})")

    print(f"\n--- Exp 6: Error analysis (two-stage BM25 k1=10) ---")
    cats = defaultdict(int)
    for r in results_per_pair:
        cats[r["error_cat"]] += 1
    total_fail = n - cats["success"]
    print(f"  success:     {cats['success']}/{n} ({cats['success']/n*100:.1f}%)")
    print(f"  stage1_miss: {cats['stage1_miss']}/{n} ({cats['stage1_miss']/n*100:.1f}%)")
    print(f"  stage2_miss: {cats['stage2_miss']}/{n} ({cats['stage2_miss']/n*100:.1f}%)")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    aggregate = {
        "n": n,
        "k": k,
        "exp2_baselines_freeform": {
            "bm25_tca5": bm25_tca,
            "minilm_tca5": min_tca,
            "e5_tca5": e5_tca,
            "bm25_mean_rank": round(sum(r["bm25_ff_rank"] for r in results_per_pair) / n, 1),
            "minilm_mean_rank": round(sum(r["minilm_ff_rank"] for r in results_per_pair) / n, 1),
        },
        "exp3_twostage_bm25_stage1": {
            f"k1_{k1}": {
                "tca5": round(sum(r["twostage"][f"bm25_k{k1}"]["tca5"] for r in results_per_pair) / n, 4),
                "mean_rank": round(sum(r["twostage"][f"bm25_k{k1}"]["rank"] for r in results_per_pair) / n, 1),
                "disc_recall_at_k1": round(sum(r["disc_recall"][f"bm25_k{k1}"] for r in results_per_pair) / n, 4),
            }
            for k1 in k1_values
        },
        "exp4_stage1_ablation": {
            s1: {
                "disc_recall_at_1": round(sum(1 for r in results_per_pair if r[f"{s1}_disc_rank"] <= 1) / n, 4),
                "disc_recall_at_5": round(sum(1 for r in results_per_pair if r[f"{s1}_disc_rank"] <= 5) / n, 4),
                "disc_recall_at_10": round(sum(1 for r in results_per_pair if r[f"{s1}_disc_rank"] <= 10) / n, 4),
                "mean_disc_rank": round(sum(r[f"{s1}_disc_rank"] for r in results_per_pair) / n, 1),
            }
            for s1 in ["bm25", "minilm", "e5"]
        },
        "exp5_hyde_freeform": {
            "tca5": round(sum(r["hyde_ff_tca5"] for r in hyde_avail) / max(len(hyde_avail), 1), 4),
            "n": len(hyde_avail),
        } if hyde_avail else None,
        "exp6_error_analysis": {
            "success": cats["success"],
            "stage1_miss": cats["stage1_miss"],
            "stage2_miss": cats["stage2_miss"],
            "total_fail": total_fail,
        },
    }

    out = {"aggregate": aggregate, "per_example": results_per_pair}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
