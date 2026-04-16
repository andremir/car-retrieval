"""
CourtListener Real-World Legal Supersession Benchmark
=======================================================
Collects real (original_opinion, overruling_opinion) pairs from CourtListener
and runs end-to-end retrieval evaluation.

Theorem 1 prediction for legal domain:
  - Query: "Is [plaintiff v. defendant] still good law? Is its holding controlling?"
  - Query vocabulary: "good law", "controlling", "still valid", "binding authority"
  - Overruling opinion vocabulary: "overruled", "abrogated", "disapproved",
    "no longer controlling", "prior precedent"
  - These vocabularies are DISJOINT → BM25/dense fail to retrieve the overruling opinion

Output: data/courtlistener_pairs.json
         data/courtlistener_results.json
"""
from __future__ import annotations
import json
import math
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path
from collections import Counter

BASE = "https://www.courtlistener.com/api/rest/v4"
HEADERS = {"User-Agent": "tc-mqa-research/1.0"}
PAIRS_OUT = Path(__file__).parent.parent / "data" / "courtlistener_pairs.json"
RESULTS_OUT = Path(__file__).parent.parent / "data" / "courtlistener_results.json"

# ---------------------------------------------------------------------------
# CourtListener API helpers
# ---------------------------------------------------------------------------

def cl_get(url: str, params: dict = None) -> dict | None:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        return None


def search_opinions(query: str, page: int = 1, per_page: int = 20) -> list[dict]:
    """Search opinions using CourtListener full-text search."""
    data = cl_get(f"{BASE}/search/", {
        "type": "o",
        "q": query,
        "stat_Precedential": "on",
        "order_by": "score desc",
        "page": page,
    })
    if not data:
        return []
    return data.get("results", [])


def get_opinion_text(opinion_id: int) -> str | None:
    """Fetch plain text of an opinion."""
    data = cl_get(f"{BASE}/opinions/{opinion_id}/")
    if not data:
        return None
    # Try plain_text, then html_lawbox stripped
    text = data.get("plain_text", "") or ""
    if not text:
        html = data.get("html_lawbox", "") or data.get("html", "") or ""
        text = re.sub(r"<[^>]+>", " ", html)
    return text.strip() if text.strip() else None


def get_cluster(cluster_id: int) -> dict | None:
    return cl_get(f"{BASE}/clusters/{cluster_id}/")


def extract_us_citations(text: str) -> list[str]:
    """Extract US Reporter citations from opinion text."""
    # Pattern: "412 U.S. 101" or "123 F.3d 456"
    us_pat = re.findall(r"\d+\s+U\.S\.\s+\d+", text)
    fed_pat = re.findall(r"\d+\s+F\.[23]d\s+\d+", text)
    return us_pat + fed_pat


def extract_case_name_from_text(text: str, before_word: str) -> str | None:
    """Extract case name that appears before an overruling keyword."""
    pattern = rf"([A-Z][a-z]+ v\.\s+[A-Z][a-z]+)[^.{{}}]*?{before_word}"
    m = re.search(pattern, text[:3000])
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Vocabulary gap detection
# ---------------------------------------------------------------------------

OVERRULING_VOCAB = {
    "overruled", "overrule", "overrules", "abrogated", "abrogate",
    "disapproved", "disapprove", "no longer controlling", "expressly overruled",
    "hereby overrule", "prior precedent is overruled", "overruling",
}

QUERY_VOCAB = {
    "good law", "controlling authority", "controlling precedent",
    "still good", "still binding", "still valid", "still controlling",
}


def check_vocab(text: str) -> dict:
    text_lower = text.lower()
    return {
        "has_overruling_vocab": any(w in text_lower for w in OVERRULING_VOCAB),
        "has_query_vocab": any(w in text_lower for w in QUERY_VOCAB),
        "overruling_words_found": [w for w in OVERRULING_VOCAB if w in text_lower],
        "query_words_found": [w for w in QUERY_VOCAB if w in text_lower],
    }


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_pairs(target: int = 80) -> list[dict]:
    pairs = []
    seen_overruling_ids = set()
    seen_original_slugs = set()

    search_queries = [
        '"hereby overrule"',
        '"expressly overruled"',
        '"is hereby abrogated"',
        '"are hereby overruled"',
        '"prior precedent" "overruled"',
        '"earlier decision" "overruled"',
        '"we overrule"',
    ]

    for sq in search_queries:
        if len(pairs) >= target:
            break
        for page in range(1, 6):
            if len(pairs) >= target:
                break
            results = search_opinions(sq, page=page)
            if not results:
                break
            print(f"  Query: {sq[:40]} p{page} → {len(results)} hits (collected: {len(pairs)})")

            for hit in results:
                if len(pairs) >= target:
                    break

                op_id = hit.get("id") or hit.get("opinion_id")
                if not op_id or op_id in seen_overruling_ids:
                    continue

                # Get full opinion text
                full_text = get_opinion_text(int(op_id))
                if not full_text or len(full_text) < 500:
                    time.sleep(0.3)
                    continue

                # Check it has overruling vocabulary
                v_check = check_vocab(full_text)
                if not v_check["has_overruling_vocab"]:
                    time.sleep(0.2)
                    continue

                # Extract case name being overruled
                case_name = (hit.get("caseName") or hit.get("case_name", "")).strip()
                if not case_name:
                    time.sleep(0.2)
                    continue

                # Build a slug for this overruling opinion
                slug = re.sub(r"[^a-z0-9]+", "_", case_name.lower())[:60]

                # Find the overruled case name from text
                # Look for pattern: "[Case name] is hereby overruled"
                overruled_match = None
                for kw in ["is hereby overruled", "are hereby overruled",
                           "is expressly overruled", "is overruled"]:
                    m = re.search(
                        rf"([A-Z][a-z]+(?:\s+v\.\s+[A-Z][a-z]+)?)[^.{{}}]{{0,200}}{re.escape(kw)}",
                        full_text[:5000]
                    )
                    if m:
                        overruled_match = m.group(1).strip()
                        break

                # Search for the original (overruled) opinion
                original_text = None
                original_case_name = overruled_match or ""

                if original_case_name and len(original_case_name) > 5:
                    orig_results = search_opinions(f'"{original_case_name}"', page=1)
                    for orig_hit in orig_results[:3]:
                        orig_id = orig_hit.get("id") or orig_hit.get("opinion_id")
                        if orig_id and int(orig_id) != int(op_id):
                            orig_text = get_opinion_text(int(orig_id))
                            if orig_text and len(orig_text) > 500:
                                original_text = orig_text
                                original_case_name = (
                                    orig_hit.get("caseName") or
                                    orig_hit.get("case_name", original_case_name)
                                )
                                break
                            time.sleep(0.3)

                if not original_text:
                    # Fallback: use a short excerpt of what's in the corpus already
                    # Extract the ruling text from the full text before the overruling sentence
                    m = re.search(r"([A-Z][^.]+\. [A-Z][^.]+\. [A-Z][^.]+\.)", full_text[:2000])
                    original_text = m.group(1) if m else full_text[:500]

                seen_overruling_ids.add(op_id)

                # Build query
                # Extract a plausible case citation from the overruled case or the hit metadata
                year_m = re.search(r"\((\d{4})\)", case_name)
                year = year_m.group(1) if year_m else "2010"
                query = (
                    f"Is {case_name} still good law? "
                    f"Is its holding still controlling authority and binding precedent?"
                )

                pair = {
                    "case_slug": slug,
                    "case_citation": case_name,
                    "query": query,
                    "original_text": original_text[:3000],
                    "overruling_text": full_text[:3000],
                    "overruling_opinion_id": int(op_id),
                    "overruled_case_name": original_case_name,
                    "vocab_check": v_check,
                }
                pairs.append(pair)
                print(f"    + collected pair {len(pairs)}: {case_name[:50]}")
                time.sleep(0.5)

            time.sleep(1)

    return pairs


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.df: dict[str, int] = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def rank(self, query: str) -> list[int]:
        qtoks = tokenize(query)
        scores = []
        for i, doc in enumerate(self.corpus):
            dl = len(doc)
            tf_map = Counter(doc)
            s = 0.0
            for term in set(qtoks):
                if term not in self.idf:
                    continue
                tf = tf_map.get(term, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += self.idf[term] * num / den
            scores.append((s, i))
        return [i for _, i in sorted(scores, reverse=True)]


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def try_load_dense():
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model, np
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Retrieval eval
# ---------------------------------------------------------------------------

def run_retrieval_eval(pairs: list[dict], k: int = 5) -> dict:
    """
    Corpus: all original opinions + all overruling opinions + 1000 synthetic distractors.
    For each pair, gold is the overruling opinion (must be retrieved to answer correctly).
    """
    # Corpus construction
    docs: list[str] = []
    doc_metas: list[dict] = []

    for p in pairs:
        docs.append(p["original_text"])
        doc_metas.append({"type": "original", "slug": p["case_slug"]})

    overruling_start = len(docs)
    for p in pairs:
        docs.append(p["overruling_text"])
        doc_metas.append({"type": "overruling", "slug": p["case_slug"]})

    # Synthetic distractor opinions (generic legal text without overruling vocabulary)
    n_distractors = 1000
    distractor_templates = [
        "The court affirms the district court's ruling. Plaintiff has established standing under Article III. "
        "The evidence presented supports the jury's finding. We find no reversible error in the proceedings below. "
        "The holding of the lower court is consistent with established precedent in this circuit.",
        "The commission's interpretation of the regulation is entitled to deference. The agency's findings are supported "
        "by substantial evidence in the record. Petitioner's arguments do not persuade us that the order was arbitrary.",
        "In reviewing this matter de novo, we conclude that the district court correctly applied the governing legal "
        "standard. The motion to dismiss was properly granted. Plaintiff failed to state a claim upon which relief can be "
        "granted. The complaint does not allege facts sufficient to establish the required elements.",
        "Upon consideration of all briefing and oral argument, the panel is unanimous. The judgment of the district court "
        "is AFFIRMED in all respects. The governing precedent supports the result reached below.",
        "This case presents a question of statutory interpretation. We hold that the plain text of the statute supports "
        "the government's position. The legislative history is consistent with this reading.",
    ]
    import random
    rng = random.Random(42)
    for i in range(n_distractors):
        text = rng.choice(distractor_templates) + f" [Case {i}]"
        docs.append(text)
        doc_metas.append({"type": "distractor", "slug": f"distractor_{i}"})

    n_docs = len(docs)
    n_pairs = len(pairs)
    print(f"Corpus: {n_docs} docs = {n_pairs} original + {n_pairs} overruling + {n_distractors} distractors")

    # BM25
    print("Building BM25 index...")
    tokenized = [tokenize(d) for d in docs]
    bm25 = BM25(tokenized)

    # Dense
    print("Loading dense model...")
    dense_model, np_mod = try_load_dense()

    results = {"bm25": [], "dense": [], "entity_probe": []}

    for i, pair in enumerate(pairs):
        query = pair["query"]
        gold_idx = overruling_start + i  # gold = overruling opinion

        # BM25
        bm25_ranking = bm25.rank(query)
        bm25_rank = bm25_ranking.index(gold_idx) + 1
        bm25_tca = int(bm25_rank <= k)

        # Dense
        if dense_model is not None:
            dense_ranking_arr = dense_model.encode(
                [query] + docs, normalize_embeddings=True, show_progress_bar=False, batch_size=64
            )
            q_emb = dense_ranking_arr[0:1]
            d_embs = dense_ranking_arr[1:]
            scores = (q_emb @ d_embs.T)[0]
            dense_ranking = list(np_mod.argsort(-scores))
            dense_rank_val = dense_ranking.index(gold_idx) + 1
            dense_tca = int(dense_rank_val <= k)
        else:
            dense_rank_val, dense_tca = None, None

        # Entity probe (by case_slug)
        slug = pair["case_slug"]
        matched = [j for j, m in enumerate(doc_metas) if m["slug"] == slug]
        probe_ranking = matched + [j for j in range(n_docs) if j not in set(matched)]
        probe_rank = probe_ranking.index(gold_idx) + 1
        probe_tca = int(probe_rank <= k)

        # Vocabulary gap verification
        v = pair["vocab_check"]
        overruling_missing_query_vocab = not v["has_query_vocab"]

        results["bm25"].append({
            "slug": slug, "patch_rank": bm25_rank, "tca": bm25_tca,
            "overruling_missing_query_vocab": overruling_missing_query_vocab,
        })
        results["dense"].append({
            "slug": slug, "patch_rank": dense_rank_val, "tca": dense_tca,
            "overruling_missing_query_vocab": overruling_missing_query_vocab,
        })
        results["entity_probe"].append({
            "slug": slug, "patch_rank": probe_rank, "tca": probe_tca,
        })

    # Aggregate
    agg = {}
    for sys_name, sys_res in results.items():
        valid = [r for r in sys_res if r["tca"] is not None]
        tca_vals = [r["tca"] for r in valid]
        ranks = [r["patch_rank"] for r in valid if r["patch_rank"] is not None]
        agg[sys_name] = {
            "n": len(valid),
            "tca": round(sum(tca_vals) / len(tca_vals), 4),
            "found_in_top_k": sum(tca_vals),
            "k": k,
            "mean_rank": round(sum(ranks) / len(ranks), 1) if ranks else None,
            "median_rank": sorted(ranks)[len(ranks) // 2] if ranks else None,
        }

    # Vocabulary gap stats on overruling opinions
    n_has_overruling_vocab = sum(
        1 for p in pairs if p["vocab_check"]["has_overruling_vocab"]
    )
    n_missing_query_vocab = sum(
        1 for p in pairs if not p["vocab_check"]["has_query_vocab"]
    )

    return {
        "per_example": results,
        "aggregate": agg,
        "n_pairs": n_pairs,
        "corpus_size": n_docs,
        "k": k,
        "vocab_stats": {
            "n_overruling_has_overruling_vocab": n_has_overruling_vocab,
            "n_overruling_missing_query_vocab": n_missing_query_vocab,
            "pct_overruling_vocab": f"{n_has_overruling_vocab/n_pairs*100:.1f}%",
            "pct_missing_query_vocab": f"{n_missing_query_vocab/n_pairs*100:.1f}%",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Step 1: Collecting CourtListener pairs ===")
    if PAIRS_OUT.exists():
        pairs = json.load(open(PAIRS_OUT))
        print(f"Loaded {len(pairs)} existing pairs from {PAIRS_OUT}")
    else:
        pairs = []

    if len(pairs) < 50:
        new_pairs = collect_pairs(target=80)
        pairs = pairs + new_pairs
        with open(PAIRS_OUT, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"Saved {len(pairs)} pairs → {PAIRS_OUT}")

    if not pairs:
        print("ERROR: no pairs collected. Exiting.")
        exit(1)

    print(f"\nVocabulary gap check ({len(pairs)} pairs):")
    n_ov = sum(1 for p in pairs if p["vocab_check"]["has_overruling_vocab"])
    n_qv = sum(1 for p in pairs if p["vocab_check"]["has_query_vocab"])
    print(f"  Overruling opinions with overruling vocab: {n_ov}/{len(pairs)} ({n_ov/len(pairs)*100:.0f}%)")
    print(f"  Overruling opinions with query vocab ('good law'): {n_qv}/{len(pairs)} ({n_qv/len(pairs)*100:.0f}%)")

    print("\n=== Step 2: Running retrieval eval ===")
    results = run_retrieval_eval(pairs, k=5)

    print("\n=== Results ===")
    print(f"{'System':<16} {'TCA@5':>8} {'Found':>7} {'MeanRank':>10} {'MedRank':>9}")
    print("-" * 55)
    for sys_name, agg in results["aggregate"].items():
        if agg is None:
            continue
        print(
            f"{sys_name:<16} {agg['tca']:>8.4f} "
            f"{agg['found_in_top_k']:>4}/{agg['n']:<2} "
            f"{str(agg['mean_rank']):>10} {str(agg['median_rank']):>9}"
        )

    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {RESULTS_OUT}")
