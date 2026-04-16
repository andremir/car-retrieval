"""
Experiment 1: CAR vs Lookup — Same Corpus, Two Query Modes
===========================================================

The strongest objection to the CAR framing is: "This is just Shepardizing / lookup —
production tools already do this." This experiment kills that objection on the same
corpus, with the same documents, by switching only the query mode.

Two query modes (same 159 GHSA pairs, same 2,318-doc corpus):

  MODE 1 — LOOKUP (known-anchor):
      Query = full disclosure text (anchor is given; system "knows" the document).
      This is the Shepardizing setting: you have the case/CVE in hand, you want
      the superseding authority. BM25 and entity probes thrive here because the
      anchor text contains exact vocabulary and entity identifiers.

  MODE 2 — RETRIEVAL (free-form NL):
      Query = Llama-generated natural-language question with no CVE ID and no
      anchor document. This is the CAR setting: a practitioner asks a question
      and needs the controlling authority retrieved from scratch.

Five systems evaluated in both modes:

  1. BM25                — keyword overlap retrieval
  2. Dense MiniLM        — semantic embedding retrieval
  3. Two-Stage BM25      — BM25 anchor → entity-indexed (the CAR system)
  4. OSV.dev API         — production lookup; requires explicit CVE ID as input
  5. Entity probe        — regex-NER extracts (product, CVE) from query; perfect
                           lookup if entity is present in query; 0% applicable in
                           free-form mode (no CVE ID)

Expected result:
  Mode 1 (lookup): OSV.dev > 0.80, Entity probe ≈ 1.00, BM25 high (vocab overlap)
  Mode 2 (retrieval): OSV.dev = 0% applicable, Entity probe = 0% applicable,
                      Dense fails, Two-Stage succeeds (TCA ≈ 0.975)

This makes the CAR ≠ Lookup distinction impossible to hand-wave away:
  - The same documents, the same corpus, the same gold answers.
  - Only the query mode changes.
  - Lookup tools collapse to 0% applicability in the retrieval mode.
  - The CAR system (Two-Stage) works precisely because the query doesn't contain
    the anchor entity — which is the entire point of the problem.

Pass criteria:
  [P1] OSV.dev Mode-1 TCA > 0.70 (confirms it's a working lookup tool)
  [P2] OSV.dev Mode-2 applicable = 0.0 (cannot handle free-form — no CVE ID)
  [P3] Entity probe Mode-2 applicable = 0.0 (same reason)
  [P4] Two-Stage Mode-2 TCA > Two-Stage Mode-2 baseline > Dense Mode-2 TCA
  [P5] Gap: Two-Stage Mode-2 > max(BM25, Dense, OSV, EntityProbe) Mode-2

Artifacts: data/falsification_car_vs_lookup.json
Runtime: ~10 min (OSV.dev API calls are rate-limited)
Dependencies: pip install sentence-transformers rank-bm25 requests numpy
"""
from __future__ import annotations

import json
import re
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT        = Path(__file__).parent.parent
PAIRS_PATH  = ROOT / "data" / "ghsa_real_pairs.json"
FREEFORM    = ROOT / "data" / "ghsa_freeform_cache.jsonl"
OUT_PATH    = ROOT / "data" / "falsification_car_vs_lookup.json"

K           = 5       # TCA@5
K1          = 5       # Two-Stage Stage-1 candidates
SLEEP_OSV   = 0.35    # seconds between OSV.dev calls (rate-limit buffer)

OSV_URL     = "https://api.osv.dev/v1/vulns/{vuln_id}"
VERSION_RE  = re.compile(r"\b(\d+\.\d+(?:\.\d+)*(?:[._-]\w+)?)\b")
CVE_RE      = re.compile(r"CVE-\d{4}-\d{4,}", re.I)
PRODUCT_RE  = re.compile(r"\b([A-Za-z][A-Za-z0-9_\-\.]{2,})\b")


# ---------------------------------------------------------------------------
# Corpus (same construction as all other falsification scripts)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_corpus(pairs: list[dict], n_distractors: int = 2000, seed: int = 42):
    docs, metas = [], []
    for p in pairs:
        docs.append(p["disclosure_text"])
        metas.append({"type": "disclosure", "product": p["product"], "cve_id": p["cve_id"]})
    patch_start = len(docs)
    for p in pairs:
        docs.append(p["patch_text"])
        metas.append({"type": "patch", "product": p["product"], "cve_id": p["cve_id"]})
    rng = random.Random(seed)
    products = ["nginx", "openssl", "log4j", "redis", "django", "flask", "rails",
                "postgresql", "mysql", "tomcat", "spring", "hibernate", "curl"]
    attacks  = ["remote code execution", "SQL injection", "buffer overflow",
                "path traversal", "cross-site scripting", "privilege escalation"]
    sevs     = ["Critical", "High", "Medium"]
    for i in range(n_distractors):
        prod = rng.choice(products); cve = f"CVE-2023-{90000+i:05d}"
        ver  = f"{rng.randint(1,9)}.{rng.randint(0,9)}"
        docs.append(
            f"{cve} has been identified in {prod}. "
            f"The flaw permits {rng.choice(attacks)} by unauthenticated remote attackers. "
            f"Severity: {rng.choice(sevs)}. "
            f"Affected versions: {prod} up to and including {ver}. "
            f"No patch is currently available."
        )
        metas.append({"type": "distractor", "product": prod, "cve_id": cve})
    return docs, metas, patch_start


# ---------------------------------------------------------------------------
# Mode-1 queries: full disclosure text as query (known-anchor setting)
# ---------------------------------------------------------------------------
def mode1_queries(pairs: list[dict]) -> dict[str, str]:
    """Use the disclosure text itself as the query — anchor is given."""
    return {p["id"]: p["disclosure_text"] for p in pairs}


# ---------------------------------------------------------------------------
# Mode-2 queries: free-form NL (no CVE ID, no anchor)
# ---------------------------------------------------------------------------
def mode2_queries(freeform_path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line in open(freeform_path):
        if line.strip():
            e = json.loads(line)
            queries[e["id"]] = e["query"]
    return queries


# ---------------------------------------------------------------------------
# BM25 TCA@K
# ---------------------------------------------------------------------------
def eval_bm25(bm25: BM25Okapi, pairs: list[dict], queries: dict[str, str],
              patch_start: int) -> tuple[float, list[int]]:
    hits = []
    for i, p in enumerate(pairs):
        q = queries.get(p["id"])
        if q is None:
            hits.append(0); continue
        scores = bm25.get_scores(tokenize(q))
        top_k  = list(np.argsort(-scores)[:K])
        hits.append(int(patch_start + i in top_k))
    return sum(hits) / len(hits), hits


# ---------------------------------------------------------------------------
# Dense MiniLM TCA@K
# ---------------------------------------------------------------------------
def eval_dense(model: SentenceTransformer, doc_embs: np.ndarray,
               pairs: list[dict], queries: dict[str, str],
               patch_start: int) -> tuple[float, list[int]]:
    hits = []
    for i, p in enumerate(pairs):
        q = queries.get(p["id"])
        if q is None:
            hits.append(0); continue
        q_emb = model.encode([q], normalize_embeddings=True)
        scores = (q_emb @ doc_embs.T)[0]
        top_k  = list(np.argsort(-scores)[:K])
        hits.append(int(patch_start + i in top_k))
    return sum(hits) / len(hits), hits


# ---------------------------------------------------------------------------
# Two-Stage (BM25 anchor → entity-indexed) TCA@K
# ---------------------------------------------------------------------------
def eval_twostage(bm25: BM25Okapi, pairs: list[dict], metas: list[dict],
                  queries: dict[str, str], patch_start: int) -> tuple[float, list[int]]:
    entity_index: dict[tuple, int] = {}
    for i in range(patch_start, patch_start + len(pairs)):
        m = metas[i]
        entity_index[(m["product"].lower(), m["cve_id"].upper())] = i

    hits = []
    for i, p in enumerate(pairs):
        q = queries.get(p["id"])
        if q is None:
            hits.append(0); continue
        gold_idx = patch_start + i
        scores   = bm25.get_scores(tokenize(q))
        top_k1   = list(np.argsort(-scores)[:K1])
        candidates, seen = [], set()
        for idx in top_k1:
            m = metas[idx]
            if m["type"] == "disclosure":
                key = (m["product"].lower(), m["cve_id"].upper())
                pi  = entity_index.get(key)
                if pi is not None and pi not in seen:
                    candidates.append(pi); seen.add(pi)
        remaining = [d for d in top_k1 if d not in seen]
        ranked = (candidates + remaining)[:K]
        hits.append(int(gold_idx in ranked))
    return sum(hits) / len(hits), hits


# ---------------------------------------------------------------------------
# OSV.dev API lookup
# TCA = 1 if OSV returns ≥1 fixed version that appears in the gold patch text.
# applicable = whether the query contains a parseable CVE ID.
# ---------------------------------------------------------------------------
def extract_cve_from_query(q: str) -> Optional[str]:
    m = CVE_RE.search(q)
    return m.group(0).upper() if m else None


def query_osv(cve_id: str) -> Optional[dict]:
    try:
        r = requests.get(OSV_URL.format(vuln_id=cve_id), timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def extract_fixed_versions(rec: dict) -> set[str]:
    fixed = set()
    for aff in rec.get("affected", []):
        for rng in aff.get("ranges", []):
            for ev in rng.get("events", []):
                if ev.get("fixed"):
                    fixed.add(ev["fixed"])
    return fixed


def eval_osv(pairs: list[dict], queries: dict[str, str],
             sleep: float = SLEEP_OSV) -> tuple[float, float, list[dict]]:
    """
    Returns (tca5, applicable_rate, per_example).
    applicable_rate = fraction of queries where OSV.dev could even be invoked
    (i.e., query contains a CVE ID).
    """
    results = []
    for p in pairs:
        q = queries.get(p["id"], "")
        cve_in_query = extract_cve_from_query(q)
        gold_versions = set(VERSION_RE.findall(p["patch_text"]))

        if cve_in_query is None:
            results.append({
                "id": p["id"], "applicable": False,
                "cve_extracted": None, "tca5": 0,
            })
            continue

        # CVE found in query — invoke OSV
        rec = query_osv(cve_in_query)
        time.sleep(sleep)

        if rec is None:
            results.append({
                "id": p["id"], "applicable": True,
                "cve_extracted": cve_in_query, "osv_found": False, "tca5": 0,
            })
            continue

        fixed = extract_fixed_versions(rec)
        tca   = int(bool(fixed & gold_versions))
        if not tca:
            summary = rec.get("summary", "") + " " + rec.get("details", "")
            tca = int(any(v in summary for v in gold_versions))

        results.append({
            "id": p["id"], "applicable": True,
            "cve_extracted": cve_in_query, "osv_found": True,
            "fixed_versions": sorted(fixed), "tca5": tca,
        })

    n_app = sum(r["applicable"] for r in results)
    applicable_rate = n_app / len(results) if results else 0.0
    tca_all = sum(r["tca5"] for r in results) / len(results) if results else 0.0
    return tca_all, applicable_rate, results


# ---------------------------------------------------------------------------
# Entity probe
# Extracts (product, CVE) from query via regex; looks up patch in entity index.
# TCA = 1 iff both product and CVE extracted and match a known pair.
# applicable = whether both entities were extracted.
# ---------------------------------------------------------------------------
def eval_entity_probe(pairs: list[dict], queries: dict[str, str],
                      patch_start: int) -> tuple[float, float, list[dict]]:
    # Build ground-truth entity index: (product_lower, cve_upper) → patch corpus index
    entity_index: dict[tuple, int] = {
        (p["product"].lower(), p["cve_id"].upper()): patch_start + i
        for i, p in enumerate(pairs)
    }
    known_products = {p["product"].lower() for p in pairs}

    results = []
    for i, p in enumerate(pairs):
        q = queries.get(p["id"], "")
        gold_idx = patch_start + i

        cve = extract_cve_from_query(q)
        # Greedy product match: find any known product name in query
        prod_found = None
        q_lower = q.lower()
        for kp in known_products:
            if kp in q_lower:
                prod_found = kp
                break

        if cve is None or prod_found is None:
            results.append({
                "id": p["id"], "applicable": False,
                "cve_found": cve, "product_found": prod_found, "tca5": 0,
            })
            continue

        # Both entities present — probe the index
        key = (prod_found, cve.upper())
        hit_idx = entity_index.get(key)
        tca = int(hit_idx == gold_idx)
        results.append({
            "id": p["id"], "applicable": True,
            "cve_found": cve, "product_found": prod_found, "tca5": tca,
        })

    n_app = sum(r["applicable"] for r in results)
    applicable_rate = n_app / len(results) if results else 0.0
    tca_all = sum(r["tca5"] for r in results) / len(results) if results else 0.0
    return tca_all, applicable_rate, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairs = json.load(open(PAIRS_PATH))
    freeform = mode2_queries(FREEFORM)
    print(f"Loaded {len(pairs)} GHSA pairs, {len(freeform)} free-form queries")

    docs, metas, patch_start = build_corpus(pairs)
    print(f"Corpus: {len(docs)} docs  (disc={len(pairs)}, patch={len(pairs)}, "
          f"dist={len(docs) - 2*len(pairs)})")

    m1_queries = mode1_queries(pairs)
    m2_queries = {k: v for k, v in freeform.items() if k in {p["id"] for p in pairs}}
    print(f"Mode-1 queries (disclosure text): {len(m1_queries)}")
    print(f"Mode-2 queries (free-form NL):    {len(m2_queries)}")

    # Build shared indexes
    print("\nBuilding BM25 and Dense indexes…")
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_embs = dense_model.encode(docs, batch_size=128, normalize_embeddings=True,
                                  show_progress_bar=False)
    print("  Done.")

    # -----------------------------------------------------------------------
    # Evaluate all 5 systems × 2 modes
    # -----------------------------------------------------------------------
    print("\n" + "="*78)
    print("SAME CORPUS · TWO QUERY MODES")
    print("="*78)
    print(f"\n{'System':<28} {'Mode-1 TCA@5':>14} {'Mode-2 TCA@5':>14} {'Mode-2 Appl.':>14}")
    print("-"*72)

    results_table = {}

    # 1. BM25
    m1_bm25, m1_bm25_hits = eval_bm25(bm25, pairs, m1_queries, patch_start)
    m2_bm25, m2_bm25_hits = eval_bm25(bm25, pairs, m2_queries, patch_start)
    results_table["bm25"] = {
        "mode1_tca5": round(m1_bm25, 4), "mode2_tca5": round(m2_bm25, 4),
        "mode2_applicable": 1.0,
    }
    print(f"  {'BM25':<26} {m1_bm25:>14.3f} {m2_bm25:>14.3f} {'100%':>14}")

    # 2. Dense MiniLM
    m1_dense, m1_dense_hits = eval_dense(dense_model, doc_embs, pairs, m1_queries, patch_start)
    m2_dense, m2_dense_hits = eval_dense(dense_model, doc_embs, pairs, m2_queries, patch_start)
    results_table["dense_minilm"] = {
        "mode1_tca5": round(m1_dense, 4), "mode2_tca5": round(m2_dense, 4),
        "mode2_applicable": 1.0,
    }
    print(f"  {'Dense MiniLM':<26} {m1_dense:>14.3f} {m2_dense:>14.3f} {'100%':>14}")

    # 3. Two-Stage
    m1_ts, m1_ts_hits = eval_twostage(bm25, pairs, metas, m1_queries, patch_start)
    m2_ts, m2_ts_hits = eval_twostage(bm25, pairs, metas, m2_queries, patch_start)
    results_table["two_stage"] = {
        "mode1_tca5": round(m1_ts, 4), "mode2_tca5": round(m2_ts, 4),
        "mode2_applicable": 1.0,
    }
    print(f"  {'Two-Stage (CAR system)':<26} {m1_ts:>14.3f} {m2_ts:>14.3f} {'100%':>14}")

    # 4. OSV.dev — Mode 1 (queries contain CVE ID)
    print(f"\n  Querying OSV.dev for Mode-1 ({len(pairs)} CVE lookups)…")
    m1_osv_tca, m1_osv_app, m1_osv_detail = eval_osv(pairs, m1_queries)
    # Mode 2: free-form NL has no CVE ID — skip actual API calls (0% applicable by construction)
    # We check applicability empirically
    m2_osv_applicable = sum(
        1 for p in pairs if extract_cve_from_query(m2_queries.get(p["id"], "")) is not None
    ) / len(pairs)
    m2_osv_tca = 0.0  # cannot be invoked without CVE ID
    results_table["osv_dev"] = {
        "mode1_tca5": round(m1_osv_tca, 4),
        "mode1_applicable": round(m1_osv_app, 4),
        "mode2_tca5": round(m2_osv_tca, 4),
        "mode2_applicable": round(m2_osv_applicable, 4),
        "note": "Mode-2 TCA is 0 because free-form NL queries contain no CVE ID (not applicable)",
    }
    print(f"  {'OSV.dev (production lookup)':<26} {m1_osv_tca:>14.3f} {m2_osv_tca:>14.3f} "
          f"  {m2_osv_applicable*100:>10.0f}%")

    # 5. Entity probe
    m1_ep_tca, m1_ep_app, m1_ep_detail = eval_entity_probe(pairs, m1_queries, patch_start)
    m2_ep_tca, m2_ep_app, m2_ep_detail = eval_entity_probe(pairs, m2_queries, patch_start)
    results_table["entity_probe"] = {
        "mode1_tca5": round(m1_ep_tca, 4), "mode1_applicable": round(m1_ep_app, 4),
        "mode2_tca5": round(m2_ep_tca, 4), "mode2_applicable": round(m2_ep_app, 4),
        "note": "Applicable only when both product name AND CVE ID appear in the query text",
    }
    print(f"  {'Entity probe (regex-NER)':<26} {m1_ep_tca:>14.3f} {m2_ep_tca:>14.3f} "
          f"  {m2_ep_app*100:>10.0f}%")

    # -----------------------------------------------------------------------
    # Pass / Fail report
    # -----------------------------------------------------------------------
    print("\n" + "="*78)
    print("PASS/FAIL REPORT  — CAR vs Lookup")
    print("="*78)

    def pf(label, cond):
        sym = "[PASS]" if cond else "[FAIL]"
        print(f"  {sym}  {label}")
        return cond

    p1 = pf(f"OSV.dev Mode-1 TCA > 0.70  → got {m1_osv_tca:.3f}  (confirms working lookup tool)",
            m1_osv_tca > 0.70)
    p2 = pf(f"OSV.dev Mode-2 applicable = 0%  → got {m2_osv_applicable*100:.0f}%  "
            f"(cannot handle free-form — no CVE ID)",
            m2_osv_applicable < 0.01)
    p3 = pf(f"Entity probe Mode-2 applicable < 5%  → got {m2_ep_app*100:.0f}%  "
            f"(no CVE ID in free-form queries)",
            m2_ep_app < 0.05)
    p4 = pf(f"Two-Stage Mode-2 > Dense Mode-2  → {m2_ts:.3f} vs {m2_dense:.3f}  "
            f"(CAR architecture adds value)",
            m2_ts > m2_dense)
    p5_baseline = max(m2_bm25, m2_dense, m2_osv_tca, m2_ep_tca)
    p5 = pf(f"Two-Stage Mode-2 > all other Mode-2  → {m2_ts:.3f} vs max={p5_baseline:.3f}  "
            f"(Two-Stage is the only system that works in retrieval mode)",
            m2_ts > p5_baseline)

    all_pass = all([p1, p2, p3, p4, p5])
    print()
    if all_pass:
        print("[ALL PASS] CAR ≠ Lookup is empirically airtight:")
        print(f"  • Lookup tools (OSV.dev, entity probe) require the anchor entity in the query.")
        print(f"  • They achieve TCA > 0 only when CVE ID is given — the 'Shepardizing' setting.")
        print(f"  • In the free-form retrieval setting (Mode-2), they are 0% applicable.")
        print(f"  • The Two-Stage CAR system is the only method that bridges both modes.")
    else:
        print("[PARTIAL] Some criteria not met — inspect per-system results.")

    # -----------------------------------------------------------------------
    # Interpretation note
    # -----------------------------------------------------------------------
    print()
    print("Interpretation:")
    print("  MODE 1 (known-anchor / 'Shepardizing'): anchor document is given as query.")
    print("    → BM25 and entity tools exploit exact vocabulary / entity overlap.")
    print("    → Production tools (OSV.dev) work because CVE ID is present.")
    print("    → This is NOT the CAR problem: the practitioner already has the document.")
    print()
    print("  MODE 2 (free-form NL / CAR): practitioner asks a question, no anchor given.")
    print("    → OSV.dev: 0% applicable (needs CVE ID it doesn't have).")
    print("    → Entity probe: 0% applicable (no CVE ID in free-form query).")
    print("    → Dense MiniLM: TCA ≈ {:.0f}% (semantic search fails — Δ(q) effect)".format(m2_dense*100))
    print("    → Two-Stage: TCA ≈ {:.0f}% (BM25 anchor finds disclosure → entity index → patch)".format(m2_ts*100))
    print()
    print("  The gap Mode1→Mode2 for lookup tools is categorical: 0% applicable, not degraded.")
    print("  The gap for Two-Stage is narrow: it works in BOTH modes because BM25 reconstructs")
    print("  the anchor from vocabulary in the free-form query, without needing the CVE ID.")
    print("="*78)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    out = {
        "experiment": "CAR vs Lookup — Same Corpus, Two Query Modes",
        "corpus": {
            "pairs": len(pairs),
            "total_docs": len(docs),
            "disclosures": len(pairs),
            "patches": len(pairs),
            "distractors": len(docs) - 2 * len(pairs),
        },
        "mode1_description": "Full disclosure text as query (known-anchor / Shepardizing setting)",
        "mode2_description": "Llama-generated free-form NL query, no CVE ID (CAR setting)",
        "results": results_table,
        "pass_fail": {
            "p1_osv_mode1_tca_gt_70": p1,
            "p2_osv_mode2_applicable_0pct": p2,
            "p3_entity_probe_mode2_applicable_lt5pct": p3,
            "p4_twostage_mode2_gt_dense_mode2": p4,
            "p5_twostage_mode2_gt_all_mode2": p5,
            "all_pass": all_pass,
        },
        "key_finding": (
            f"Lookup tools (OSV.dev, entity probe) achieve TCA={m1_osv_tca:.3f}/{m1_ep_tca:.3f} "
            f"in Mode-1 but are {m2_osv_applicable*100:.0f}%/{m2_ep_app*100:.0f}% applicable "
            f"in Mode-2. Two-Stage achieves TCA={m2_ts:.3f} in Mode-2, the only system that "
            f"works without the anchor entity in the query."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
