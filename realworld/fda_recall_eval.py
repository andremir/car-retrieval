"""
FDA Drug Recall Benchmark: medical/regulatory temporal supersession retrieval.

Pairs: (approved drug label, FDA enforcement/recall notice)
  - Anchor: drug label with clinical indications — "indicated for treatment of type 2 diabetes"
  - Superseder: recall enforcement notice — "recalled due to NDMA impurity, lot #12345"
  - Vocabulary gap: queries use clinical language; recall notices use administrative language

Entity key: generic_name (normalized lowercase)

Pipeline:
  Stage 1 (BM25): clinical query → retrieve the drug's approved label (anchor)
  Stage 2 (entity index): generic_name → retrieve the recall notice (superseder)

Source:
  OpenFDA drug enforcement API: api.fda.gov/drug/enforcement.json
  OpenFDA drug label API:       api.fda.gov/drug/label.json

Output: data/fda_recall_pairs.json
        data/fda_recall_results.json
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.request
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent
PAIRS_PATH       = ROOT / "data" / "fda_recall_pairs.json"
LABEL_CACHE      = ROOT / "data" / "fda_label_cache.json"
ENFREC_CACHE     = ROOT / "data" / "fda_enforcement_cache.json"
OUT_PATH         = ROOT / "data" / "fda_recall_results.json"

OPENFDA_BASE = "https://api.fda.gov"
K  = 5
K1 = 5
N_DISTRACTORS = 2000
MAX_PAIRS = 500
BATCH = 100     # OpenFDA page size

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fda_get(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tc-mqa-research/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception:
        return None


def clean_text(text: str) -> str:
    """Remove section numbers, extra whitespace from FDA label text."""
    text = re.sub(r"^\d+(\.\d+)?\s+[A-Z ]+\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_first_indication(text: str, drug_name: str) -> str:
    """Return first 200 chars of indication text, stripped of headers."""
    t = clean_text(text)
    # Remove common header patterns
    for pat in [r"INDICATIONS AND USAGE", r"^\d\s+INDICATION", r"1 INDICATION"]:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
    # Drop drug trade name at start
    if t.upper().startswith(drug_name.upper()[:8]):
        t = t[len(drug_name):].strip().lstrip(",. ")
    return t[:250].strip()


def normalize_name(name: str) -> str:
    """Normalize drug name for matching."""
    name = name.lower()
    name = re.sub(r"\s+(hydrochloride|hcl|sulfate|phosphate|sodium|acetate|tartrate|"
                  r"mesylate|maleate|fumarate|chloride|bromide|citrate|succinate)$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Step 1: Collect enforcement records
# ---------------------------------------------------------------------------

def fetch_enforcement_records(max_pairs: int = MAX_PAIRS) -> list[dict]:
    """Fetch drug enforcement records that have openfda.generic_name."""
    if ENFREC_CACHE.exists():
        records = json.load(open(ENFREC_CACHE))
        print(f"Loaded {len(records)} cached enforcement records")
        return records

    records = []
    skip = 0
    print("Fetching FDA enforcement records...")
    while len(records) < max_pairs * 6:  # over-fetch since ~21% have generic_name
        url = (f"{OPENFDA_BASE}/drug/enforcement.json"
               f"?search=product_type:Drugs&limit={BATCH}&skip={skip}")
        data = fda_get(url)
        if not data:
            break
        batch = data.get("results", [])
        if not batch:
            break
        for r in batch:
            openfda = r.get("openfda", {})
            generic_names = openfda.get("generic_name", [])
            if not generic_names:
                continue
            # Use first generic name (simplest)
            generic = generic_names[0]
            norm = normalize_name(generic)
            if len(norm) < 3:
                continue
            records.append({
                "generic_name": generic,
                "norm_name": norm,
                "brand_name": (openfda.get("brand_name") or [generic])[0],
                "product_description": r.get("product_description", "")[:600],
                "reason_for_recall": r.get("reason_for_recall", "")[:400],
                "recall_number": r.get("recall_number", ""),
                "recall_date": r.get("recall_initiation_date", ""),
                "classification": r.get("classification", ""),
            })
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        skip += BATCH
        if skip >= total:
            break
        time.sleep(0.3)  # rate limiting
        if skip % 1000 == 0:
            print(f"  Fetched {skip}/{total}, collected {len(records)} with generic_name")

    print(f"Collected {len(records)} enforcement records with generic_name")
    with open(ENFREC_CACHE, "w") as f:
        json.dump(records, f)
    print(f"Saved enforcement cache → {ENFREC_CACHE}")
    return records


# ---------------------------------------------------------------------------
# Step 2: Fetch drug labels
# ---------------------------------------------------------------------------

def fetch_label(norm_name: str, cache: dict) -> dict | None:
    """Fetch drug label for normalized generic name (with cache)."""
    if norm_name in cache:
        return cache[norm_name]

    # Try exact search first
    for search_name in [norm_name, norm_name.split()[0]]:
        url = (f"{OPENFDA_BASE}/drug/label.json"
               f"?search=openfda.generic_name:{urllib.parse.quote(search_name)}&limit=1")
        data = fda_get(url)
        if data and data.get("results"):
            result = data["results"][0]
            label = {
                "indications_and_usage": (result.get("indications_and_usage") or [""])[0],
                "description": (result.get("description") or [""])[0],
                "warnings": (result.get("warnings") or
                             result.get("warnings_and_cautions") or [""])[0],
            }
            if len(label["indications_and_usage"]) > 50:
                cache[norm_name] = label
                time.sleep(0.25)
                return label
        time.sleep(0.25)

    cache[norm_name] = None
    return None


# ---------------------------------------------------------------------------
# Step 3: Build pairs
# ---------------------------------------------------------------------------

def build_pairs(records: list[dict], label_cache: dict,
                max_pairs: int = MAX_PAIRS) -> list[dict]:
    """Match enforcement records to drug labels, deduplicate by norm_name."""
    seen_names: set[str] = set()
    pairs = []

    print(f"Building pairs (target: {max_pairs})...")
    for i, rec in enumerate(records):
        if len(pairs) >= max_pairs:
            break
        norm = rec["norm_name"]
        if norm in seen_names:
            continue

        label = fetch_label(norm, label_cache)
        if not label:
            continue
        indication = label["indications_and_usage"]
        if len(indication) < 60:
            continue

        seen_names.add(norm)

        # Build condition-only query: no drug name, only clinical indication vocabulary.
        # This enforces the vocabulary gap: the query uses therapeutic/disease language,
        # while recall notices use administrative language (lot numbers, contamination).
        # Dense retrieval cannot bridge this gap without drug-name as anchor.
        indication_excerpt = extract_first_indication(indication, rec["generic_name"])
        # Strip drug name variants from the excerpt so query is truly name-free
        for variant in [rec["generic_name"], rec["brand_name"].split(",")[0].strip(),
                        norm, norm.split()[0]]:
            indication_excerpt = re.sub(re.escape(variant), "this drug",
                                        indication_excerpt, flags=re.IGNORECASE)
        indication_excerpt = indication_excerpt[:180].rstrip(". ")
        query = (
            f"Is the drug indicated for the following still currently approved "
            f"and safe, or has FDA regulatory action changed its status? "
            f"{indication_excerpt}?"
        )

        # Anchor text = label (clinical language)
        anchor_text = (
            f"{rec['generic_name']} - Approved Drug Label. "
            f"{clean_text(indication)[:800]} "
            f"{clean_text(label['description'])[:300]}"
        )

        # Superseder text = recall notice (administrative language)
        superseder_text = (
            f"FDA Drug Enforcement Action. "
            f"Product: {rec['product_description']}. "
            f"Reason for recall: {rec['reason_for_recall']}. "
            f"Recall number: {rec['recall_number']}. "
            f"Classification: {rec['classification']}."
        )

        pairs.append({
            "id": f"fda_{len(pairs):04d}",
            "norm_name": norm,
            "generic_name": rec["generic_name"],
            "brand_name": rec["brand_name"],
            "query": query,
            "anchor_text": anchor_text,
            "superseder_text": superseder_text,
            "recall_date": rec["recall_date"],
            "reason_for_recall": rec["reason_for_recall"][:200],
        })
        if len(pairs) % 50 == 0 or len(pairs) <= 3:
            print(f"  [{len(pairs):>3}] {norm}: {len(indication)} chars indication")

    print(f"Built {len(pairs)} pairs")
    return pairs


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

STOP = {"a", "an", "the", "is", "in", "of", "for", "to", "and", "or", "as",
        "be", "that", "this", "with", "it", "at", "by", "on", "are", "was",
        "from", "has", "have", "been", "which", "may", "not", "its", "were"}

def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\b[a-z]{2,}\b", text.lower()) if w not in STOP]


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.df: Counter = Counter()
        self.tf: list[Counter] = []
        for doc in corpus:
            self.tf.append(Counter(doc))
            for term in set(doc):
                self.df[term] += 1
        self.idf: dict[str, float] = {
            t: math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for t, df in self.df.items()
        }
        self._dls = [len(d) for d in corpus]
        # Inverted index for efficiency
        self._inv: dict[str, list[int]] = {}
        for i, doc in enumerate(corpus):
            for term in set(doc):
                self._inv.setdefault(term, []).append(i)

    def rank(self, query: str, top_n: int | None = None) -> list[int]:
        qtoks = list(set(tokenize(query)))
        candidate_docs: set[int] = set()
        for t in qtoks:
            candidate_docs.update(self._inv.get(t, []))
        scores: dict[int, float] = {}
        for i in candidate_docs:
            dl = self._dls[i]
            s = 0.0
            for term in qtoks:
                tf = self.tf[i].get(term, 0)
                idf = self.idf.get(term, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * num / den
            scores[i] = s
        ranked = sorted(candidate_docs, key=lambda i: -scores.get(i, 0.0))
        # Append unscored docs at end
        ranked_set = set(ranked)
        ranked.extend(i for i in range(self.n) if i not in ranked_set)
        if top_n:
            return ranked[:top_n]
        return ranked


# ---------------------------------------------------------------------------
# Corpus + Evaluation
# ---------------------------------------------------------------------------

DISTRACTOR_TEMPLATES = [
    "GENERIC DRUG - APPROVED DRUG LABEL. {drug} is indicated for the treatment of {cond} "
    "in adults and pediatric patients. The recommended dosage is {dose} mg orally twice daily. "
    "Common adverse reactions include nausea and headache.",
    "FDA DRUG ENFORCEMENT ACTION. Product: {drug} {form} tablets, {qty} count bottles. "
    "Reason for recall: {reason}. Recall number: D-{num}. Classification: Class II.",
]

# Use synthetic drug names (not real) so distractors cannot overlap with real pairs
DRUGS  = ["Xeltroven", "Cardivexin", "Nephrolan", "Glibecoxil", "Thyravast",
          "Losparex", "Azibactin", "Gabavexin", "Sertivalen", "Hydrotrizan",
          "Prevocillin", "Benzavexin", "Cortavelin", "Metavoxin", "Primocillin",
          "Quinavexin", "Renavolin", "Silbavexin", "Temovaxin", "Urovastein"]
CONDS  = ["hypertension", "hyperlipidemia", "acid reflux", "hypothyroidism", "anxiety"]
FORMS  = ["immediate release", "extended release", "delayed release"]
REAS   = ["Subpotent product", "Failed dissolution testing", "Contamination with particulate matter",
          "Out-of-specification stability results", "Mislabeling", "Cross-contamination"]

import random


def build_corpus(pairs: list[dict]) -> tuple[list[str], list[dict], int, int]:
    docs, metas = [], []
    for p in pairs:
        docs.append(p["anchor_text"])
        metas.append({"type": "label", "norm_name": p["norm_name"]})

    recall_start = len(docs)
    for p in pairs:
        docs.append(p["superseder_text"])
        metas.append({"type": "recall", "norm_name": p["norm_name"]})

    rng = random.Random(42)
    for i in range(N_DISTRACTORS):
        drug = rng.choice(DRUGS)
        tmpl = rng.choice(DISTRACTOR_TEMPLATES)
        text = tmpl.format(
            drug=drug, cond=rng.choice(CONDS), dose=rng.randint(5, 500),
            form=rng.choice(FORMS), qty=rng.randint(30, 500),
            reason=rng.choice(REAS), num=90000 + i
        )
        docs.append(text)
        metas.append({"type": "distractor", "norm_name": f"distractor_{i}"})

    return docs, metas, 0, recall_start


def run_eval(pairs: list[dict]) -> dict:
    docs, metas, label_start, recall_start = build_corpus(pairs)
    n_pairs = len(pairs)
    print(f"Corpus: {len(docs)} docs "
          f"({n_pairs} labels + {n_pairs} recalls + {N_DISTRACTORS} distractors)")

    # BM25
    print("Building BM25 index...")
    bm25 = BM25([tokenize(d) for d in docs])

    # Dense
    print("Loading MiniLM...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Encoding corpus...")
    doc_embs = model.encode(docs, batch_size=128, normalize_embeddings=True,
                            show_progress_bar=True)

    # Entity index: norm_name → recall index
    entity_index: dict[str, int] = {}
    for j in range(recall_start, recall_start + n_pairs):
        entity_index[metas[j]["norm_name"]] = j

    queries = [p["query"] for p in pairs]
    print("Encoding queries...")
    q_embs = model.encode(queries, batch_size=128, normalize_embeddings=True,
                          show_progress_bar=False)

    bm25_direct_hits  = 0
    dense_direct_hits = 0
    twostage_hits     = 0
    stage1_hits       = 0  # did BM25 retrieve the label in top-k1?

    per_example = []
    print("Evaluating...")
    for i, (pair, q_emb) in enumerate(zip(pairs, q_embs)):
        gold_idx   = recall_start + i   # correct answer = recall notice
        anchor_idx = label_start  + i   # anchor = drug label

        norm = pair["norm_name"]

        # BM25 direct
        bm25_ranking = bm25.rank(pair["query"])
        bm25_rank = bm25_ranking.index(gold_idx) + 1
        bm25_tca  = int(bm25_rank <= K)
        bm25_direct_hits += bm25_tca

        # Dense direct
        scores = doc_embs @ q_emb
        dense_ranking = list(np.argsort(-scores))
        dense_rank = dense_ranking.index(gold_idx) + 1
        dense_tca  = int(dense_rank <= K)
        dense_direct_hits += dense_tca

        # Two-stage: BM25 Stage-1 → entity index Stage-2
        bm25_top_k1 = bm25_ranking[:K1]
        stage1_found = anchor_idx in bm25_top_k1
        stage1_hits += int(stage1_found)

        if stage1_found:
            # Promote recall notice to position determined by anchor's rank
            anchor_rank_in_top = bm25_top_k1.index(anchor_idx)
            # Final ranking: superseder at position anchor_rank_in_top (0-indexed)
            ts_rank = anchor_rank_in_top + 1
        else:
            # Anchor not in Stage-1 → fall back to BM25 rank of the recall directly
            ts_rank = bm25_rank  # will be > K in most cases
        twostage_tca = int(ts_rank <= K)
        twostage_hits += twostage_tca

        per_example.append({
            "id": pair["id"],
            "norm_name": norm,
            "bm25_rank": bm25_rank, "bm25_tca5": bm25_tca,
            "dense_rank": dense_rank, "dense_tca5": dense_tca,
            "stage1_found": stage1_found,
            "twostage_rank": ts_rank, "twostage_tca5": twostage_tca,
        })

    n = len(pairs)
    agg = {
        "n": n, "k": K, "k1": K1, "corpus_size": len(docs),
        "n_distractors": N_DISTRACTORS,
        "bm25_direct_tca5":  round(bm25_direct_hits  / n, 4),
        "dense_minilm_tca5": round(dense_direct_hits  / n, 4),
        "twostage_tca5":     round(twostage_hits       / n, 4),
        "stage1_anchor_recall5": round(stage1_hits / n, 4),
    }
    return {"aggregate": agg, "per_example": per_example}


# ---------------------------------------------------------------------------
# Vocabulary gap verification
# ---------------------------------------------------------------------------

CLINICAL_VOCAB = {
    "indicated", "indication", "treatment", "therapy", "patients", "disease",
    "condition", "disorder", "symptoms", "clinical", "efficacy", "approved",
    "dosage", "diagnosis", "therapeutic",
}
ADMIN_VOCAB = {
    "recall", "lot", "sterility", "contamination", "impurity", "dissolution",
    "specification", "potency", "mislabeling", "adulterated", "misbranded",
    "subpotent", "particulate",
}

def check_vocab_gap(pairs: list[dict]) -> dict:
    n = len(pairs)
    recall_has_clinical = sum(
        1 for p in pairs
        if any(w in p["superseder_text"].lower() for w in CLINICAL_VOCAB)
    )
    recall_has_admin = sum(
        1 for p in pairs
        if any(w in p["superseder_text"].lower() for w in ADMIN_VOCAB)
    )
    label_has_clinical = sum(
        1 for p in pairs
        if any(w in p["anchor_text"].lower() for w in CLINICAL_VOCAB)
    )
    return {
        "n": n,
        "recall_pct_with_clinical_vocab": round(recall_has_clinical / n, 3),
        "recall_pct_with_admin_vocab":    round(recall_has_admin    / n, 3),
        "label_pct_with_clinical_vocab":  round(label_has_clinical  / n, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load label cache
    if LABEL_CACHE.exists():
        label_cache: dict = json.load(open(LABEL_CACHE))
        print(f"Loaded {len(label_cache)} cached labels")
    else:
        label_cache = {}

    # Load or build pairs
    if PAIRS_PATH.exists():
        pairs = json.load(open(PAIRS_PATH))
        print(f"Loaded {len(pairs)} existing pairs")
    else:
        pairs = []

    if len(pairs) < MAX_PAIRS:
        records = fetch_enforcement_records(max_pairs=MAX_PAIRS)
        pairs = build_pairs(records, label_cache, max_pairs=MAX_PAIRS)
        with open(PAIRS_PATH, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"Saved {len(pairs)} pairs → {PAIRS_PATH}")
        with open(LABEL_CACHE, "w") as f:
            json.dump(label_cache, f)
        print(f"Saved label cache ({len(label_cache)} entries) → {LABEL_CACHE}")

    # Vocabulary gap analysis
    vgap = check_vocab_gap(pairs)
    print(f"\nVocabulary gap analysis ({vgap['n']} pairs):")
    print(f"  Recall notices with clinical vocab:  {vgap['recall_pct_with_clinical_vocab']*100:.1f}%")
    print(f"  Recall notices with admin vocab:     {vgap['recall_pct_with_admin_vocab']*100:.1f}%")
    print(f"  Drug labels with clinical vocab:     {vgap['label_pct_with_clinical_vocab']*100:.1f}%")

    # Run evaluation
    print(f"\nRunning retrieval evaluation on {len(pairs)} pairs...")
    results = run_eval(pairs)
    results["vocab_gap"] = vgap

    agg = results["aggregate"]
    print(f"\n=== FDA Drug Recall Results (n={agg['n']}, k={agg['k']}, "
          f"corpus={agg['corpus_size']}) ===")
    print(f"BM25 direct TCA@5:            {agg['bm25_direct_tca5']:.4f}")
    print(f"Dense (MiniLM) direct TCA@5:  {agg['dense_minilm_tca5']:.4f}")
    print(f"Two-Stage TCA@5:              {agg['twostage_tca5']:.4f}")
    print(f"Stage-1 anchor recall@5:      {agg['stage1_anchor_recall5']:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
