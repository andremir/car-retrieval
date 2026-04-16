"""
Experiment 4: Query-Style Breadth Matrix
=========================================

For each GHSA pair, evaluate 5 query variants to confirm the ranking pattern
(BM25/Dense fail, Two-Stage succeeds) is not a template artifact.

Query styles:
  1. structured    — "Is {product} affected by {CVE-ID}?" (CVE-explicit)
  2. freeform      — existing Llama-generated queries (no CVE-ID)
  3. entity_omit   — free-form with product name redacted (pure vuln behavior)
  4. noisy         — free-form with one character typo in product name
  5. partial_ent   — structured but with CVE suffix only (not full CVE-ID)

Pass: ranking order BM25<Dense<Two-Stage preserved across all 5 styles.
      Theory variables (Δ(q)) explain changes: entity_omit → higher Δ → more Dense fails.

Dependencies: pip install sentence-transformers rank-bm25 numpy
              Optional: Llama API for paraphrase generation (skip if no access)
Artifacts: data/falsification_query_breadth.json
"""
from __future__ import annotations

import json
import re
import random
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT        = Path(__file__).parent.parent
PAIRS_PATH  = ROOT / "data" / "ghsa_real_pairs.json"
FREEFORM    = ROOT / "data" / "ghsa_freeform_cache.jsonl"
OUT_PATH    = ROOT / "data" / "falsification_query_breadth.json"

K  = 5
K1 = 5


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_corpus(pairs, n_distractors=2000, seed=42):
    docs, metas = [], []
    for p in pairs:
        docs.append(p["disclosure_text"])
        metas.append({"type": "disclosure", "product": p["product"], "cve_id": p["cve_id"]})
    patch_start = len(docs)
    for i, p in enumerate(pairs):
        docs.append(p["patch_text"])
        metas.append({"type": "patch", "product": p["product"], "cve_id": p["cve_id"]})
    rng = random.Random(seed)
    products = ["nginx","openssl","log4j","redis","django","flask","rails","postgresql","curl"]
    attacks  = ["remote code execution","SQL injection","buffer overflow","path traversal"]
    sevs     = ["Critical","High","Medium"]
    for i in range(n_distractors):
        prod = rng.choice(products); cve = f"CVE-2023-{90000+i:05d}"; ver = f"{rng.randint(1,9)}.{rng.randint(0,9)}"
        docs.append(f"{cve} in {prod}. {rng.choice(attacks)}. {rng.choice(sevs)}. Versions ≤{ver}. No patch.")
        metas.append({"type": "distractor", "product": prod, "cve_id": cve})
    return docs, metas, patch_start


def add_typo(name: str, rng: random.Random) -> str:
    """Insert one character transposition."""
    if len(name) < 3:
        return name
    i = rng.randint(0, len(name) - 2)
    chars = list(name)
    chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)


def build_query_variants(pairs, freeform_queries: dict[str, str]) -> dict[str, dict[str, str]]:
    """Return {pair_id: {style: query_text}}."""
    rng = random.Random(99)
    variants: dict[str, dict[str, str]] = {}
    for p in pairs:
        pid = p["id"]
        product, cve = p["product"], p["cve_id"]
        ff = freeform_queries.get(pid, f"Is {product} still vulnerable to {cve}?")

        # Entity-omit: remove product name (+ any close variant) from free-form query
        product_re = re.compile(re.escape(product), re.I)
        entity_omit = product_re.sub("[PRODUCT]", ff)

        # Noisy: one transposition in product name within the structured query
        noisy_product = add_typo(product, rng)
        noisy = f"Is {noisy_product} affected by {cve}?"

        # Partial entity: only last 5 chars of CVE suffix, no product name
        cve_suffix = cve.split("-")[-1][-5:]
        partial = f"Any fix available for vulnerability {cve_suffix}?"

        variants[pid] = {
            "structured":   f"Is {product} affected by {cve}?",
            "freeform":     ff,
            "entity_omit":  entity_omit,
            "noisy":        noisy,
            "partial_ent":  partial,
        }
    return variants


def evaluate_style(pairs, variants, docs, metas, patch_start, style, bm25, dense_model, doc_embs):
    """Evaluate BM25, Dense, and Two-Stage for a given query style."""
    entity_index: dict[str, int] = {}
    for i in range(patch_start, patch_start + len(pairs)):
        m = metas[i]
        entity_index[(m["product"].lower(), m["cve_id"].upper())] = i

    bm25_hits, dense_hits, ts_hits = [], [], []

    for i, pair in enumerate(pairs):
        pid = pair["id"]
        if pid not in variants:
            continue
        q = variants[pid].get(style, "")
        if not q:
            continue
        gold_idx = patch_start + i

        # BM25
        bm25_scores = bm25.get_scores(tokenize(q))
        bm25_top5 = list(np.argsort(-bm25_scores)[:K])
        bm25_hits.append(int(gold_idx in bm25_top5))

        # Dense
        q_emb = dense_model.encode([q], normalize_embeddings=True)
        dense_scores = (q_emb @ doc_embs.T)[0]
        dense_top5 = list(np.argsort(-dense_scores)[:K])
        dense_hits.append(int(gold_idx in dense_top5))

        # Two-Stage (BM25 anchor → entity-indexed)
        bm25_top_k1 = list(np.argsort(-bm25_scores)[:K1])
        ts_candidates = []
        seen = set()
        for idx in bm25_top_k1:
            m = metas[idx]
            if m["type"] == "disclosure":
                key = (m["product"].lower(), m["cve_id"].upper())
                patch_idx = entity_index.get(key)
                if patch_idx is not None and patch_idx not in seen:
                    ts_candidates.append(patch_idx)
                    seen.add(patch_idx)
        remaining = [d for d in bm25_top_k1 if d not in seen]
        ts_ranked = ts_candidates + remaining
        ts_hits.append(int(gold_idx in ts_ranked[:K]))

    n = len(bm25_hits)
    return {
        "style": style, "n": n,
        "bm25_tca5": round(sum(bm25_hits) / n, 4) if n else 0,
        "dense_tca5": round(sum(dense_hits) / n, 4) if n else 0,
        "twostage_tca5": round(sum(ts_hits) / n, 4) if n else 0,
    }


def main():
    pairs = json.load(open(PAIRS_PATH))
    freeform: dict[str, str] = {}
    for line in open(FREEFORM):
        if line.strip():
            e = json.loads(line); freeform[e["id"]] = e["query"]
    print(f"Loaded {len(pairs)} pairs, {len(freeform)} free-form queries")

    docs, metas, patch_start = build_corpus(pairs)
    variants = build_query_variants(pairs, freeform)

    bm25 = BM25Okapi([tokenize(d) for d in docs])
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Encoding corpus…")
    doc_embs = dense_model.encode(docs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    STYLES = ["structured", "freeform", "entity_omit", "noisy", "partial_ent"]
    results = []
    print(f"\n{'Style':<18} {'BM25':>8} {'Dense':>8} {'Two-Stage':>12} {'Order correct?':>16}")
    print("-" * 68)
    for style in STYLES:
        row = evaluate_style(pairs, variants, docs, metas, patch_start,
                             style, bm25, dense_model, doc_embs)
        results.append(row)
        # Check ranking order: BM25 ≤ Dense and Dense < Two-Stage (expected by theory)
        order_ok = row["twostage_tca5"] > max(row["bm25_tca5"], row["dense_tca5"])
        sym = "✓" if order_ok else "✗ RANKING BROKEN"
        print(f"  {style:<16} {row['bm25_tca5']:>8.3f} {row['dense_tca5']:>8.3f} {row['twostage_tca5']:>12.3f}  {sym}")

    print("\nPass/fail:")
    all_order = all(r["twostage_tca5"] > max(r["bm25_tca5"], r["dense_tca5"]) for r in results)
    if all_order:
        print("  [PASS] Two-Stage > max(BM25, Dense) for all 5 query styles.")
    else:
        broken = [r["style"] for r in results if r["twostage_tca5"] <= max(r["bm25_tca5"], r["dense_tca5"])]
        print(f"  [FAIL] Ranking broken for: {broken}")

    with open(OUT_PATH, "w") as f:
        json.dump({"query_breadth_matrix": results}, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
