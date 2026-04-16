"""
Public legal end-to-end benchmark on SCOTUS overruling pairs from Cornell LII.

Builds a public subset of curated overruling pairs using Cornell Legal
Information Institute (LII) full-text pages, then evaluates direct retrieval
 versus a two-stage controlling-authority pipeline.

Query style:
  "Is {original_case} still good law and controlling precedent?"

Corpus:
  - original opinion text for each usable pair
  - overruling opinion text for each usable pair

Each document is prefixed with its own case name and citation, matching how
legal search systems index title metadata.

Outputs:
  data/legal_lii_pairs.json
  data/legal_lii_results.json
"""
from __future__ import annotations

import json
import re
import statistics
import urllib.request
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
SOURCE_PAIRS = ROOT / "data" / "legal_real_pairs.json"
PAIRS_OUT = ROOT / "data" / "legal_lii_pairs.json"
RESULTS_OUT = ROOT / "data" / "legal_lii_results.json"

HEADERS = {"User-Agent": "car-retrieval/1.0"}
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"


def citation_to_lii_url(citation: str) -> str | None:
    match = re.search(r"(\d+)\s+U\.S\.\s+(\d+)", citation)
    if not match:
        return None
    volume, page = match.groups()
    return f"https://www.law.cornell.edu/supremecourt/text/{volume}/{page}"


def fetch_lii_text(citation: str) -> tuple[str, str]:
    root_url = citation_to_lii_url(citation)
    if root_url is None:
        raise ValueError(f"Unsupported citation: {citation}")

    root_html = urllib.request.urlopen(
        urllib.request.Request(root_url, headers=HEADERS),
        timeout=20,
    ).read().decode("utf-8", "ignore")

    links = sorted(
        set(
            re.findall(
                r"""href=["'](/supremecourt/text/\d+/\d+/USSC_PRO_[^"']+)["']""",
                root_html,
            )
        )
    )
    if not links:
        links = [re.sub(r"https://www\.law\.cornell\.edu", "", root_url)]

    texts: list[str] = []
    for link in links:
        page_url = f"https://www.law.cornell.edu{link}" if link.startswith("/") else link
        html = urllib.request.urlopen(
            urllib.request.Request(page_url, headers=HEADERS),
            timeout=20,
        ).read().decode("utf-8", "ignore")
        soup = BeautifulSoup(html, "html.parser")
        main = soup.select_one("main")
        text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        texts.append(text)

    return root_url, re.sub(r"\s+", " ", " ".join(texts)).strip()


def build_public_pairs() -> list[dict]:
    raw_pairs = json.loads(SOURCE_PAIRS.read_text())
    public_pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for pair in raw_pairs:
        key = (pair["original_citation"], pair["overruling_citation"])
        if key in seen:
            continue
        seen.add(key)

        if pair["original_citation"] == pair["overruling_citation"]:
            continue

        try:
            original_url, original_text = fetch_lii_text(pair["original_citation"])
            overruling_url, overruling_text = fetch_lii_text(pair["overruling_citation"])
        except Exception:
            continue

        slug = re.sub(r"[^a-z0-9]+", "_", pair["original_case"].lower()).strip("_")
        public_pairs.append(
            {
                "id": pair["id"],
                "case_slug": slug,
                "query": f"Is {pair['original_case']} still good law and controlling precedent?",
                "original_case": pair["original_case"],
                "original_citation": pair["original_citation"],
                "original_url": original_url,
                "overruling_case": pair["overruling_case"],
                "overruling_citation": pair["overruling_citation"],
                "overruling_url": overruling_url,
                "original_text": (
                    f"Case: {pair['original_case']}. "
                    f"Citation: {pair['original_citation']}. "
                    f"{original_text}"
                ),
                "overruling_text": (
                    f"Case: {pair['overruling_case']}. "
                    f"Citation: {pair['overruling_citation']}. "
                    f"{overruling_text}"
                ),
            }
        )

    return public_pairs


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / max(self.n, 1)
        self.df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.idf = {
            term: np.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        scores: list[float] = []
        for doc in self.corpus:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            dl = len(doc)
            score = 0.0
            for term in set(q_terms):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores.append(score)
        return scores


def rank_to_metrics(rank: int, k: int) -> tuple[int, int]:
    return int(rank <= k), rank


def evaluate(pairs: list[dict], k: int = 5) -> dict:
    corpus = [pair["original_text"] for pair in pairs] + [pair["overruling_text"] for pair in pairs]
    n = len(pairs)

    bm25 = BM25([tokenize(doc) for doc in corpus])
    dense = SentenceTransformer(MINILM_ID)
    dense_embeddings = dense.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    query_embeddings = dense.encode(
        [pair["query"] for pair in pairs],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dense_scores = query_embeddings @ dense_embeddings.T

    rows: list[dict] = []
    for idx, pair in enumerate(pairs):
        original_idx = idx
        overruling_idx = n + idx

        bm25_ranking = np.argsort(-np.asarray(bm25.scores(pair["query"])))
        dense_ranking = np.argsort(-dense_scores[idx])

        bm25_orig_rank = int(np.where(bm25_ranking == original_idx)[0][0]) + 1
        bm25_over_rank = int(np.where(bm25_ranking == overruling_idx)[0][0]) + 1
        dense_orig_rank = int(np.where(dense_ranking == original_idx)[0][0]) + 1
        dense_over_rank = int(np.where(dense_ranking == overruling_idx)[0][0]) + 1

        rows.append(
            {
                "id": pair["id"],
                "original_case": pair["original_case"],
                "overruling_case": pair["overruling_case"],
                "query": pair["query"],
                "bm25_direct_rank": bm25_over_rank,
                "bm25_direct_tca5": int(bm25_over_rank <= k),
                "bm25_anchor_rank": bm25_orig_rank,
                "bm25_anchor_recall5": int(bm25_orig_rank <= k),
                "two_stage_bm25_rank": bm25_orig_rank,
                "two_stage_bm25_tca5": int(bm25_orig_rank <= k),
                "dense_direct_rank": dense_over_rank,
                "dense_direct_tca5": int(dense_over_rank <= k),
                "dense_anchor_rank": dense_orig_rank,
                "dense_anchor_recall5": int(dense_orig_rank <= k),
                "two_stage_dense_rank": dense_orig_rank,
                "two_stage_dense_tca5": int(dense_orig_rank <= k),
            }
        )

    def mean(field: str) -> float:
        return round(statistics.mean(row[field] for row in rows), 3)

    def median(field: str) -> float:
        return round(statistics.median(row[field] for row in rows), 3)

    aggregate = {
        "n_pairs": n,
        "corpus_size": len(corpus),
        "query_template": "Is {original_case} still good law and controlling precedent?",
        "document_indexing_note": (
            "Each opinion is prefixed with its own case name and citation, matching "
            "standard legal metadata indexing."
        ),
        "bm25_direct_tca5": mean("bm25_direct_tca5"),
        "bm25_direct_mean_rank": mean("bm25_direct_rank"),
        "bm25_direct_median_rank": median("bm25_direct_rank"),
        "bm25_anchor_recall_at_5": mean("bm25_anchor_recall5"),
        "two_stage_bm25_tca5": mean("two_stage_bm25_tca5"),
        "dense_minilm_direct_tca5": mean("dense_direct_tca5"),
        "dense_minilm_direct_mean_rank": mean("dense_direct_rank"),
        "dense_minilm_direct_median_rank": median("dense_direct_rank"),
        "dense_minilm_anchor_recall_at_5": mean("dense_anchor_recall5"),
        "two_stage_dense_minilm_tca5": mean("two_stage_dense_tca5"),
        "model_id": MINILM_ID,
    }

    return {"aggregate": aggregate, "rows": rows}


def main() -> None:
    public_pairs = build_public_pairs()
    PAIRS_OUT.write_text(json.dumps(public_pairs, indent=2))
    results = evaluate(public_pairs, k=5)
    RESULTS_OUT.write_text(json.dumps(results, indent=2))

    agg = results["aggregate"]
    print(f"Public SCOTUS pairs: {agg['n_pairs']}")
    print(f"Corpus size: {agg['corpus_size']}")
    print(
        "BM25 direct TCA@5:",
        f"{agg['bm25_direct_tca5']:.3f}",
        "| Two-stage BM25 TCA@5:",
        f"{agg['two_stage_bm25_tca5']:.3f}",
    )
    print(
        "Dense direct TCA@5:",
        f"{agg['dense_minilm_direct_tca5']:.3f}",
        "| Two-stage dense TCA@5:",
        f"{agg['two_stage_dense_minilm_tca5']:.3f}",
    )
    print(f"Wrote {PAIRS_OUT}")
    print(f"Wrote {RESULTS_OUT}")


if __name__ == "__main__":
    main()
