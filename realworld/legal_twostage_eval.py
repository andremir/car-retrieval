"""
Two-stage pipeline evaluation on LegalPrecedentQA T1 examples.

Uses the generic CAR framework (supersession_rag.py).

For each T1 example:
  - Generate a free-form query (no case citation) via Llama 3.3 70B
  - Stage 1: BM25 retrieves ruling_issued from the free-form query
  - Stage 2: entity-indexed lookup (case_slug) → ruling_overruled

Corpus: 250 ruling_issued + 250 ruling_overruled + 1000 synthetic distractors

Output: data/legal_twostage_results.json
"""
from __future__ import annotations
import os
import json, re, time, random, sys, urllib.request
from pathlib import Path
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from supersession_rag import (
    Record, LegalAdapter, make_pipeline, _tokenize, _BM25
)

# ---------------------------------------------------------------------------
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_URL = "https://api.studio.nebius.ai/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

ROOT = Path(__file__).parent.parent
LEGAL_DATA = ROOT / "data" / "legalprecedentqa_v1.jsonl"
CACHE_PATH = ROOT / "data" / "legal_freeform_cache.jsonl"
OUT_PATH   = ROOT / "data" / "legal_twostage_results.json"

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
FREEFORM_PROMPT = """\
Here is a legal opinion:

{ruling_text}

Write ONE question (maximum 50 words) that a lawyer or researcher would ask to find out whether the legal holding in this opinion has been reversed or overruled by a later court.

Rules:
- Do NOT include any full case citation (e.g. "Nelson v. Perez, 330 U.S. 625")
- Do NOT use the words "overrule", "abrogate", or "reverse"
- Describe the legal topic in natural language
- End with a question mark

Output only the question, nothing else."""


def llm_complete(prompt: str, retries: int = 3) -> str | None:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.2,
    }).encode()
    headers = {"Authorization": f"Bearer {NEBIUS_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(NEBIUS_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None


# ---------------------------------------------------------------------------
# Distractor corpus
# ---------------------------------------------------------------------------
DISTRACTOR_TEMPLATES = [
    "The court affirms the district court's ruling. Plaintiff has established standing under Article III. "
    "The evidence presented supports the jury's finding. We find no reversible error in the proceedings below.",
    "The commission's interpretation is entitled to deference. The agency's findings are supported by "
    "substantial evidence. Petitioner's arguments do not persuade us the order was arbitrary or capricious.",
    "Upon consideration of all briefing and oral argument, the panel is unanimous. The judgment of the "
    "district court is AFFIRMED in all respects. The governing precedent supports the result reached below.",
    "This case presents a question of statutory interpretation. We hold that the plain text of the statute "
    "supports the government's position. The legislative history is consistent with this reading.",
    "In reviewing this matter de novo, the district court correctly applied the governing legal standard. "
    "The motion to dismiss was properly granted. Plaintiff failed to state a claim upon which relief can be granted.",
    "The Fourth Amendment requires a warrant before law enforcement may search a suspect's home. "
    "Here, no warrant was obtained. The evidence must therefore be suppressed under the exclusionary rule.",
    "Contract formation requires offer, acceptance, and consideration. The trial court correctly found "
    "that no valid contract was formed because the parties never reached a meeting of the minds.",
    "Trademark infringement requires proof of likelihood of confusion. The Polaroid factors weigh against "
    "the plaintiff on balance. The district court's denial of injunctive relief is affirmed.",
]


def build_corpus(examples: list[dict], n_distractors: int = 1000) -> list[Record]:
    rng = random.Random(42)
    records = []

    # Ruling issued (the "anchor" documents)
    for ex in examples:
        kb = ex["kb"]
        issued = next((e for e in kb if e["event_type"] == "ruling_issued"), None)
        if issued:
            records.append(Record(
                record_type="ruling_issued",
                timestamp=datetime.fromisoformat(issued["timestamp"]),
                text=issued["text"],
                entities={"case_slug": ex["employee"]},
            ))

    # Ruling overruled (gold documents)
    for ex in examples:
        kb = ex["kb"]
        overruled = next((e for e in kb if e["event_type"] == "ruling_overruled"), None)
        if overruled:
            records.append(Record(
                record_type="ruling_overruled",
                timestamp=datetime.fromisoformat(overruled["timestamp"]),
                text=overruled["text"],
                entities={"case_slug": ex["employee"]},
            ))

    # Distractors
    for i in range(n_distractors):
        text = rng.choice(DISTRACTOR_TEMPLATES) + f" [Docket {1000+i}]"
        records.append(Record(
            record_type="distractor",
            timestamp=datetime(2024, 1, 1),
            text=text,
            entities={"case_slug": f"distractor_{i}"},
        ))

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load T1 examples
    lines = open(LEGAL_DATA).readlines()
    all_examples = [json.loads(l) for l in lines]
    t1_examples = [e for e in all_examples if "type1" in e["hop_type"]]
    print(f"T1 examples: {len(t1_examples)}")

    # Load free-form cache
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["id"]] = entry["query"]
        print(f"Loaded {len(cache)} cached free-form queries")

    # Generate free-form queries
    new_entries = []
    CITE_RE = re.compile(r"\b[A-Z][a-z]+ v\.\s+[A-Z][a-z]+\b|CVE-\d{4}-\d+", re.IGNORECASE)

    for i, ex in enumerate(t1_examples):
        eid = ex["example_id"]
        if eid in cache:
            continue
        kb = ex["kb"]
        issued = next((e for e in kb if e["event_type"] == "ruling_issued"), None)
        if not issued:
            continue

        prompt = FREEFORM_PROMPT.format(ruling_text=issued["text"][:600])
        query = llm_complete(prompt)
        if not query:
            continue

        query = query.strip('"\'').strip()
        # Strip case citations if LLM included them
        if CITE_RE.search(query):
            query = CITE_RE.sub("[case]", query).strip()

        cache[eid] = query
        new_entries.append({"id": eid, "query": query})

        if (i + 1) % 50 == 0 or i < 3:
            print(f"  [{i+1:>3}/{len(t1_examples)}] {query[:70]}")

        time.sleep(0.2)

    if new_entries:
        with open(CACHE_PATH, "a") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")
        print(f"Saved {len(new_entries)} new queries → {CACHE_PATH}")

    # Build corpus
    corpus = build_corpus(t1_examples)
    n_issued = sum(1 for r in corpus if r.record_type == "ruling_issued")
    n_over   = sum(1 for r in corpus if r.record_type == "ruling_overruled")
    overruled_start = n_issued
    print(f"\nCorpus: {len(corpus)} records ({n_issued} issued + {n_over} overruled + distractors)")

    # Build pipeline with LegalAdapter
    adapter = LegalAdapter()
    pipeline = make_pipeline(corpus, adapter)

    # Also build separate BM25 for baseline
    bm25_standalone = _BM25([_tokenize(r.text) for r in corpus])

    # Evaluate
    k, k1 = 5, 5
    results = []

    for i, ex in enumerate(t1_examples):
        eid = ex["example_id"]
        ff_query = cache.get(eid)
        if not ff_query:
            continue

        gold_idx = overruled_start + i  # ruling_overruled for this example

        # BM25 baseline
        bm25_scores = bm25_standalone.scores(ff_query)
        bm25_ranking = sorted(range(len(corpus)), key=lambda j: -bm25_scores[j])
        bm25_rank = bm25_ranking.index(gold_idx) + 1

        # Anchor (ruling_issued) BM25 rank
        anchor_idx = i  # ruling_issued for this example
        anchor_rank = bm25_ranking.index(anchor_idx) + 1

        # Two-stage
        ts_ranking = pipeline.retrieve(ff_query, k1=k1)
        ts_rank = ts_ranking.index(gold_idx) + 1

        results.append({
            "id": eid,
            "case_slug": ex["employee"],
            "freeform_query": ff_query,
            "bm25_rank": bm25_rank,
            "bm25_tca5": int(bm25_rank <= k),
            "anchor_bm25_rank": anchor_rank,
            "anchor_in_top5": int(anchor_rank <= k),
            "twostage_rank": ts_rank,
            "twostage_tca5": int(ts_rank <= k),
        })

        if (i + 1) % 50 == 0 or i < 3:
            print(f"  [{i+1:>3}/{len(t1_examples)}] bm25={bm25_rank:>5} ts={ts_rank:>3} "
                  f"anchor@5={'Y' if anchor_rank<=k else 'N'}")

    n = len(results)
    print(f"\n=== Legal Two-Stage Results (n={n}, k={k}) ===")
    bm25_tca  = sum(r["bm25_tca5"] for r in results) / n
    ts_tca    = sum(r["twostage_tca5"] for r in results) / n
    anchor_r1 = sum(1 for r in results if r["anchor_bm25_rank"] <= 1) / n
    anchor_r5 = sum(1 for r in results if r["anchor_bm25_rank"] <= k) / n
    print(f"BM25 TCA@5:      {bm25_tca:.4f}")
    print(f"Two-Stage TCA@5: {ts_tca:.4f}")
    print(f"Stage-1 Anchor Recall@1: {anchor_r1:.4f}")
    print(f"Stage-1 Anchor Recall@5: {anchor_r5:.4f}")

    # Identity: TCA@k = anchor_recall@k (should hold)
    print(f"TCA@5 == Anchor@5? {abs(ts_tca - anchor_r5) < 0.001}")

    agg = {
        "n": n, "k": k, "k1": k1,
        "corpus_size": len(corpus),
        "bm25_tca5": round(bm25_tca, 4),
        "twostage_tca5": round(ts_tca, 4),
        "stage1_anchor_recall_at_1": round(anchor_r1, 4),
        "stage1_anchor_recall_at_5": round(anchor_r5, 4),
    }
    out = {"aggregate": agg, "per_example": results}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
