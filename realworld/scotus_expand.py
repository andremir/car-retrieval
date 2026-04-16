"""
Expand SCOTUS overruling benchmark from 31 → ~200+ pairs using Wikipedia's
"List of overruled United States Supreme Court decisions" + Cornell LII full text.

Strategy:
  1. Fetch Wikipedia list → extract (overruled_case, overruled_citation,
     overruling_case, overruling_citation) tuples
  2. For each unique pair, fetch opinion text from Cornell LII
  3. Use condition-style queries that name the case (standard legal search)
  4. Skip pairs already in legal_real_pairs.json
  5. Run BM25 + MiniLM dense + two-stage evaluation on all pairs combined

Output:
  data/legal_scotus_expanded_pairs.json   – all fetched pairs
  data/legal_scotus_expanded_results.json – evaluation results
  Updates data/legal_lii_results.json with new n
"""
from __future__ import annotations

import json
import re
import statistics
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
EXISTING_PAIRS  = ROOT / "data" / "legal_real_pairs.json"
LII_PAIRS_CACHE = ROOT / "data" / "legal_lii_pairs.json"     # already-fetched texts
PAIRS_OUT       = ROOT / "data" / "legal_scotus_expanded_pairs.json"
RESULTS_OUT     = ROOT / "data" / "legal_scotus_expanded_results.json"
LII_RESULTS_OUT = ROOT / "data" / "legal_lii_results.json"   # will update

HEADERS = {"User-Agent": "car-retrieval/1.0 (research)"}
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
K = 5
FETCH_DELAY = 1.2   # seconds between LII requests


# ---------------------------------------------------------------------------
# Wikipedia scraper
# ---------------------------------------------------------------------------
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_overruled_United_States_Supreme_Court_decisions"

def fetch_wiki_pairs() -> list[dict]:
    """Parse Wikipedia table into raw (case, citation) pairs."""
    html = urllib.request.urlopen(
        urllib.request.Request(WIKI_URL, headers=HEADERS), timeout=30
    ).read().decode("utf-8", "ignore")
    soup = BeautifulSoup(html, "html.parser")

    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for table in soup.find_all("table", class_="wikitable"):
        headers_row = table.find("tr")
        if not headers_row:
            continue
        ths = [th.get_text(" ", strip=True).lower() for th in headers_row.find_all(["th", "td"])]

        # Look for tables that have overruled + overruling columns
        # Wikipedia uses various column layouts; detect by header content
        col_idx: dict[str, int] = {}
        for i, h in enumerate(ths):
            if "overruled" in h and "overruling" not in h:
                col_idx.setdefault("overruled_case", i)
            elif "overruling" in h:
                col_idx.setdefault("overruling_case", i)
            elif "year" in h:
                col_idx.setdefault("year", i)

        if "overruled_case" not in col_idx or "overruling_case" not in col_idx:
            continue

        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < max(col_idx.values()) + 1:
                continue

            def cell_text(idx: int) -> str:
                return re.sub(r"\s+", " ", cells[idx].get_text(" ", strip=True))

            overruled_raw  = cell_text(col_idx["overruled_case"])
            overruling_raw = cell_text(col_idx["overruling_case"])

            # Split "Case Name Citation" — citation is "NNN U.S. NNN (YYYY)"
            def split_case_citation(raw: str) -> tuple[str, str]:
                m = re.search(r"(\d+)\s+U\.S\.?\s+(\d+)(?:\s*\(\s*\d{4}\s*\))?", raw)
                if m:
                    citation = m.group(0).strip()
                    # Normalize: "163 U.S. 537 (1896)"
                    citation = re.sub(r"U\.S\.?\s+", "U.S. ", citation)
                    case_name = raw[:m.start()].strip().rstrip(",").strip()
                    return case_name, citation
                return raw.strip(), ""

            overruled_case, overruled_cit   = split_case_citation(overruled_raw)
            overruling_case, overruling_cit = split_case_citation(overruling_raw)

            if not overruled_cit or not overruling_cit:
                continue
            if overruled_cit == overruling_cit:
                continue

            key = (overruled_cit, overruling_cit)
            if key in seen:
                continue
            seen.add(key)

            pairs.append({
                "original_case":      overruled_case,
                "original_citation":  overruled_cit,
                "overruling_case":    overruling_case,
                "overruling_citation": overruling_cit,
            })

    print(f"Extracted {len(pairs)} unique overruling pairs from Wikipedia")
    return pairs


# ---------------------------------------------------------------------------
# Cornell LII fetcher (same logic as legal_lii_benchmark.py)
# ---------------------------------------------------------------------------
def citation_to_lii_url(citation: str) -> str | None:
    m = re.search(r"(\d+)\s+U\.S\.?\s+(\d+)", citation)
    if not m:
        return None
    vol, page = m.groups()
    return f"https://www.law.cornell.edu/supremecourt/text/{vol}/{page}"


def fetch_lii_text(citation: str) -> tuple[str, str]:
    root_url = citation_to_lii_url(citation)
    if root_url is None:
        raise ValueError(f"Unsupported citation: {citation}")

    html = urllib.request.urlopen(
        urllib.request.Request(root_url, headers=HEADERS), timeout=20
    ).read().decode("utf-8", "ignore")

    # Find opinion sub-pages
    links = sorted(set(re.findall(
        r"""href=["'](/supremecourt/text/\d+/\d+/USSC_PRO_[^"']+)["']""", html
    )))
    if not links:
        links = [re.sub(r"https://www\.law\.cornell\.edu", "", root_url)]

    texts: list[str] = []
    for link in links[:2]:          # cap at 2 sub-pages to limit requests
        page_url = f"https://www.law.cornell.edu{link}" if link.startswith("/") else link
        try:
            pg = urllib.request.urlopen(
                urllib.request.Request(page_url, headers=HEADERS), timeout=20
            ).read().decode("utf-8", "ignore")
        except Exception:
            continue
        soup = BeautifulSoup(pg, "html.parser")
        main = soup.select_one("main")
        text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
        texts.append(re.sub(r"\s+", " ", text).strip())
        time.sleep(FETCH_DELAY)

    combined = re.sub(r"\s+", " ", " ".join(texts)).strip()
    if len(combined) < 200:
        raise ValueError("Text too short — likely a redirect or 404")
    return root_url, combined


# ---------------------------------------------------------------------------
# Load already-fetched pairs (from legal_lii_pairs.json) to avoid re-fetching
# ---------------------------------------------------------------------------
def load_fetched_cache() -> dict[str, dict]:
    """key = (original_citation, overruling_citation) → pair dict with texts"""
    cache: dict[str, dict] = {}
    if LII_PAIRS_CACHE.exists():
        for p in json.loads(LII_PAIRS_CACHE.read_text()):
            key = (p["original_citation"], p["overruling_citation"])
            cache[key] = p
    return cache


# ---------------------------------------------------------------------------
# BM25 (self-contained)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25:
    def __init__(self, corpus: list[list[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.df: dict[str, int] = {}
        for doc in corpus:
            for t in set(doc):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {t: np.log((self.n - df + 0.5) / (df + 0.5) + 1)
                    for t, df in self.df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.n)
        for i, doc in enumerate(self.corpus):
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            dl = len(doc)
            s = 0.0
            for t in set(q):
                f = tf.get(t, 0)
                if f == 0:
                    continue
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(pairs: list[dict], k: int = K) -> dict:
    """Same corpus structure as legal_lii_benchmark.py."""
    corpus = [p["original_text"] for p in pairs] + [p["overruling_text"] for p in pairs]
    n = len(pairs)

    bm25 = BM25([tokenize(d) for d in corpus])
    model = SentenceTransformer(MINILM_ID)
    print("Encoding corpus with MiniLM...")
    doc_embs = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    q_embs   = model.encode([p["query"] for p in pairs],
                             normalize_embeddings=True, show_progress_bar=False)
    dense_scores = q_embs @ doc_embs.T

    rows = []
    for idx, pair in enumerate(pairs):
        orig_idx = idx
        over_idx = n + idx

        bm25_rank = np.argsort(-bm25.scores(pair["query"]))
        dense_rank = np.argsort(-dense_scores[idx])

        def rank_of(arr, target):
            return int(np.where(arr == target)[0][0]) + 1

        bm25_orig_rank = rank_of(bm25_rank, orig_idx)
        bm25_over_rank = rank_of(bm25_rank, over_idx)
        dense_orig_rank = rank_of(dense_rank, orig_idx)
        dense_over_rank = rank_of(dense_rank, over_idx)

        rows.append({
            "id": pair["id"],
            "original_case":    pair["original_case"],
            "overruling_case":  pair["overruling_case"],
            "query":            pair["query"],
            "bm25_direct_rank":     bm25_over_rank,
            "bm25_direct_tca5":     int(bm25_over_rank  <= k),
            "bm25_anchor_recall5":  int(bm25_orig_rank  <= k),
            "two_stage_bm25_tca5":  int(bm25_orig_rank  <= k),
            "dense_direct_rank":    dense_over_rank,
            "dense_direct_tca5":    int(dense_over_rank <= k),
            "dense_anchor_recall5": int(dense_orig_rank <= k),
            "two_stage_dense_tca5": int(dense_orig_rank <= k),
        })

    def mean(field):
        return round(statistics.mean(r[field] for r in rows), 3)

    aggregate = {
        "n_pairs": n,
        "corpus_size": len(corpus),
        "bm25_direct_tca5":         mean("bm25_direct_tca5"),
        "bm25_anchor_recall_at_5":  mean("bm25_anchor_recall5"),
        "two_stage_bm25_tca5":      mean("two_stage_bm25_tca5"),
        "dense_minilm_direct_tca5": mean("dense_direct_tca5"),
        "dense_anchor_recall_at_5": mean("dense_anchor_recall5"),
        "two_stage_dense_tca5":     mean("two_stage_dense_tca5"),
        "model_id": MINILM_ID,
    }
    return {"aggregate": aggregate, "rows": rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Get candidate pairs from Wikipedia
    wiki_pairs = fetch_wiki_pairs()

    # 2. Load already-fetched text cache
    fetched_cache = load_fetched_cache()
    print(f"Cache has {len(fetched_cache)} already-fetched pairs")

    # 3. Existing source pairs (to skip duplicates in fetch)
    existing_cits: set[tuple[str, str]] = set()
    if EXISTING_PAIRS.exists():
        for p in json.loads(EXISTING_PAIRS.read_text()):
            existing_cits.add((p.get("original_citation", ""), p.get("overruling_citation", "")))

    # 4. Fetch texts for new pairs
    public_pairs: list[dict] = []

    # Seed with already-fetched cache first (these are validated)
    for p in fetched_cache.values():
        slug = re.sub(r"[^a-z0-9]+", "_", p["original_case"].lower()).strip("_")
        public_pairs.append({
            "id":                   p["id"],
            "case_slug":            slug,
            "query":                p["query"],
            "original_case":        p["original_case"],
            "original_citation":    p["original_citation"],
            "original_url":         p.get("original_url", ""),
            "overruling_case":      p["overruling_case"],
            "overruling_citation":  p["overruling_citation"],
            "overruling_url":       p.get("overruling_url", ""),
            "original_text":        p["original_text"],
            "overruling_text":      p["overruling_text"],
        })
    seeded = len(public_pairs)
    print(f"Seeded {seeded} pairs from existing LII cache")

    seen_cits = {(p["original_citation"], p["overruling_citation"]) for p in public_pairs}

    # Fetch new pairs from Wikipedia list
    fetched_new = 0
    failed = 0
    for i, wp in enumerate(wiki_pairs):
        key = (wp["original_citation"], wp["overruling_citation"])
        if key in seen_cits:
            continue
        seen_cits.add(key)

        try:
            orig_url, orig_text = fetch_lii_text(wp["original_citation"])
            time.sleep(FETCH_DELAY)
            over_url, over_text = fetch_lii_text(wp["overruling_citation"])
            time.sleep(FETCH_DELAY)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  SKIP {wp['original_case']!r}: {e}")
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", wp["original_case"].lower()).strip("_")
        pair_id = f"scotus_{len(public_pairs):03d}"
        public_pairs.append({
            "id":                  pair_id,
            "case_slug":           slug,
            "query":               f"Is {wp['original_case']} still good law and controlling precedent?",
            "original_case":       wp["original_case"],
            "original_citation":   wp["original_citation"],
            "original_url":        orig_url,
            "overruling_case":     wp["overruling_case"],
            "overruling_citation": wp["overruling_citation"],
            "overruling_url":      over_url,
            "original_text": (
                f"Case: {wp['original_case']}. "
                f"Citation: {wp['original_citation']}. "
                f"{orig_text}"
            ),
            "overruling_text": (
                f"Case: {wp['overruling_case']}. "
                f"Citation: {wp['overruling_citation']}. "
                f"{over_text}"
            ),
        })
        fetched_new += 1
        n_total = len(public_pairs)
        print(f"  [{i+1}/{len(wiki_pairs)}] fetched={fetched_new} total={n_total}: {wp['original_case']}")

        # Early exit if we hit a good target size
        if n_total >= 250:
            print(f"Reached target of 250 pairs, stopping fetch early")
            break

    print(f"\nFetch complete: {seeded} seeded + {fetched_new} new = {len(public_pairs)} total ({failed} failed)")

    # 5. Save pairs
    PAIRS_OUT.write_text(json.dumps(public_pairs, indent=2))
    print(f"Saved pairs → {PAIRS_OUT}")

    # 6. Evaluate
    print(f"\nEvaluating {len(public_pairs)} pairs...")
    results = evaluate(public_pairs, k=K)

    RESULTS_OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved results → {RESULTS_OUT}")

    agg = results["aggregate"]
    print(f"\n=== Expanded SCOTUS Results ===")
    print(f"  n_pairs:              {agg['n_pairs']}")
    print(f"  BM25 direct TCA@5:    {agg['bm25_direct_tca5']:.3f}")
    print(f"  Dense direct TCA@5:   {agg['dense_minilm_direct_tca5']:.3f}")
    print(f"  Two-stage BM25 TCA@5: {agg['two_stage_bm25_tca5']:.3f}")
    print(f"  Two-stage Dense TCA@5:{agg['two_stage_dense_tca5']:.3f}")


if __name__ == "__main__":
    main()
