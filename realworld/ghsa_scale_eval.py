"""
Exp A: Scale amplifies the error.

Tests multiple embedding models (different parameter counts) on the 159 real GHSA
corpus. Hypothesis: larger / semantically richer models achieve higher disclosure
recall@5 (they match the query-to-disclosure vocabulary better), but LOWER TCA@5
(they rank the semantically-similar disclosure above the patch, making TCA worse).

Models tested (single family, three sizes, to isolate scale vs. architecture):
  E5 family (intfloat):
    e5-small-v2  ~  33 M params
    e5-base-v2   ~ 109 M params
    e5-large-v2  ~ 335 M params
  BGE family (BAAI) — second family to confirm the pattern is cross-architecture:
    bge-small-en-v1.5  ~  33 M
    bge-base-en-v1.5   ~ 109 M
    bge-large-en-v1.5  ~ 335 M
  Plus the existing MiniLM baseline for reference.

Output: data/ghsa_scale_results.json
"""
from __future__ import annotations
import json, re, random, time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT       = Path(__file__).parent.parent
PAIRS_PATH = ROOT / "data" / "ghsa_real_pairs.json"
FREEFORM   = ROOT / "data" / "ghsa_freeform_cache.jsonl"
OUT_PATH   = ROOT / "data" / "ghsa_scale_results.json"

# (model_id, param_count_M, display_name, family, needs_query_prefix)
MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2",        22,  "MiniLM-L6",    "MiniLM", False),
    ("intfloat/e5-small-v2",                           33,  "E5-small",     "E5",     True),
    ("intfloat/e5-base-v2",                           109,  "E5-base",      "E5",     True),
    ("intfloat/e5-large-v2",                          335,  "E5-large",     "E5",     True),
    ("BAAI/bge-small-en-v1.5",                         33,  "BGE-small",    "BGE",    False),
    ("BAAI/bge-base-en-v1.5",                         109,  "BGE-base",     "BGE",    False),
    ("BAAI/bge-large-en-v1.5",                        335,  "BGE-large",    "BGE",    False),
]

N_DISTRACTORS = 2000
K = 5


def tokenize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_corpus(pairs, n_distractors=N_DISTRACTORS):
    docs, metas = [], []
    disc_start = 0
    for p in pairs:
        docs.append(p["disclosure_text"])
        metas.append({"type": "disclosure", "product": p["product"],
                      "cve_id": p["cve_id"], "pair_i": len(metas)})
    patch_start = len(docs)
    for i, p in enumerate(pairs):
        docs.append(p["patch_text"])
        metas.append({"type": "patch", "product": p["product"],
                      "cve_id": p["cve_id"], "pair_i": i})
    rng = random.Random(42)
    products = ["nginx","openssl","log4j","redis","django","flask","rails",
                "postgresql","mysql","tomcat","spring","curl","libssl","libjpeg","zlib"]
    for i in range(n_distractors):
        product = rng.choice(products)
        cve = f"CVE-2023-{90000+i:05d}"
        docs.append(f"{cve} has been identified in {product}. "
                    f"The flaw permits remote code execution. "
                    f"Severity: High. No patch currently available.")
        metas.append({"type": "distractor", "product": product, "cve_id": cve, "pair_i": -1})
    return docs, metas, disc_start, patch_start


def build_entity_index(metas, patch_start):
    idx = {}
    for i in range(patch_start, len(metas)):
        m = metas[i]
        if m["type"] == "patch":
            key = (m["product"].lower(), m["cve_id"].upper())
            idx[key] = i
    return idx


def embed_corpus(model: SentenceTransformer, texts: list[str],
                 needs_prefix: bool, prefix: str = "passage: ") -> np.ndarray:
    """Embed corpus documents (with passage prefix if needed)."""
    if needs_prefix:
        texts = [prefix + t for t in texts]
    return model.encode(texts, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)


def embed_queries(model: SentenceTransformer, queries: list[str],
                  needs_prefix: bool, prefix: str = "query: ") -> np.ndarray:
    if needs_prefix:
        queries = [prefix + q for q in queries]
    return model.encode(queries, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True)


def eval_model(model_id: str, param_m: int, display: str,
               needs_prefix: bool,
               docs: list[str], metas: list[dict],
               disc_start: int, patch_start: int,
               entity_index: dict, queries: list[str],
               gold_patch_idxs: list[int]) -> dict:
    print(f"\n  Loading {display} ({param_m}M)...")
    t0 = time.time()
    model = SentenceTransformer(model_id)

    doc_embs  = embed_corpus(model, docs, needs_prefix)
    q_embs    = embed_queries(model, queries, needs_prefix)
    load_time = time.time() - t0
    print(f"  Encoded in {load_time:.1f}s")

    tca_hits = 0
    disc_hits = 0
    patch_direct_hits = 0

    for i, (q_emb, gold_patch_idx) in enumerate(zip(q_embs, gold_patch_idxs)):
        scores = doc_embs @ q_emb
        ranking = np.argsort(-scores).tolist()
        top_k = ranking[:K]

        # TCA@5: is the gold patch in top-5?
        if gold_patch_idx in top_k:
            tca_hits += 1

        # Disclosure recall@5: is the disclosure in top-5?
        disc_idx = disc_start + i
        if disc_idx in top_k:
            disc_hits += 1

        # Patch direct (what happens if we just do straight retrieval)
        if gold_patch_idx in ranking[:K]:
            patch_direct_hits += 1

    n = len(queries)
    result = {
        "model":        display,
        "model_id":     model_id,
        "params_M":     param_m,
        "tca5":         round(tca_hits / n, 4),
        "disc_recall5": round(disc_hits / n, 4),
        "n":            n,
    }
    print(f"  TCA@5={result['tca5']:.4f}  Disc@5={result['disc_recall5']:.4f}")
    return result


def main():
    pairs = json.load(open(PAIRS_PATH))
    print(f"Loaded {len(pairs)} pairs")

    # Load free-form queries
    freeform: dict[str, str] = {}
    for line in open(FREEFORM):
        if line.strip():
            e = json.loads(line)
            freeform[e["id"]] = e["query"]

    # Filter to pairs that have a query
    eval_pairs = [p for p in pairs if p["id"] in freeform]
    queries = [freeform[p["id"]] for p in eval_pairs]
    print(f"Eval pairs with free-form queries: {len(eval_pairs)}")

    # Build corpus from ALL pairs (same as twostage_eval)
    docs, metas, disc_start, patch_start = build_corpus(pairs)
    entity_index = build_entity_index(metas, patch_start)
    print(f"Corpus: {len(docs)} docs ({patch_start} disclosures, "
          f"{patch_start - disc_start} patches + distractors)")

    # For each eval pair, the gold patch is at patch_start + pair_index_in_all_pairs
    pair_id_to_all_idx = {p["id"]: i for i, p in enumerate(pairs)}
    gold_patch_idxs = [patch_start + pair_id_to_all_idx[p["id"]] for p in eval_pairs]
    # disc_start offsets for eval pairs
    disc_eval_idxs  = [disc_start  + pair_id_to_all_idx[p["id"]] for p in eval_pairs]

    results = []
    for model_id, param_m, display, family, needs_prefix in MODELS:
        try:
            r = eval_model(
                model_id=model_id, param_m=param_m, display=display,
                needs_prefix=needs_prefix,
                docs=docs, metas=metas,
                disc_start=disc_start, patch_start=patch_start,
                entity_index=entity_index,
                queries=queries, gold_patch_idxs=gold_patch_idxs,
            )
            r["family"] = family
            results.append(r)
        except Exception as e:
            print(f"  FAILED {display}: {e}")

    # Sort by params for readability
    results.sort(key=lambda r: r["params_M"])

    print("\n\n=== Scale Amplifies Error ===")
    print(f"{'Model':<16} {'Params':>8}  {'Disc@5':>7}  {'TCA@5':>7}")
    print("-" * 46)
    for r in results:
        print(f"{r['model']:<16} {r['params_M']:>8}M  {r['disc_recall5']:>7.4f}  {r['tca5']:>7.4f}")

    out = {"n_eval": len(eval_pairs), "k": K, "results": results}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
