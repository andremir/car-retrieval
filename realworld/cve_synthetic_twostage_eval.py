"""
Two-stage pipeline evaluation on CVEPatchQA T1 synthetic examples.

Uses the CAR framework with SecurityAdapter.

For each T1 example:
  - Generate a free-form query (no CVE ID) via Llama 3.3 70B
  - Stage 1: BM25 retrieves cve_disclosed from the free-form query
  - Stage 2: entity-indexed lookup (product, cve_id) → patch_released

Output: data/cve_synthetic_twostage_results.json
"""
from __future__ import annotations
import os
import json, re, time, random, sys, urllib.request
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from supersession_rag import (
    Record, SecurityAdapter, make_pipeline, _tokenize, _BM25
)

# ---------------------------------------------------------------------------
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_URL = "https://api.studio.nebius.ai/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

ROOT = Path(__file__).parent.parent
CVE_DATA  = ROOT / "data" / "cvepatchqa_v1.jsonl"
CACHE_PATH = ROOT / "data" / "cve_synthetic_freeform_cache.jsonl"
OUT_PATH   = ROOT / "data" / "cve_synthetic_twostage_results.json"

FREEFORM_PROMPT = """\
Here is a security advisory:

{disclosure_text}

Write ONE question (maximum 50 words) that a developer would ask to find out whether this vulnerability has been patched in the software.

Rules:
- Do NOT include any CVE identifier (e.g. CVE-2024-XXXXX)
- Describe the specific vulnerability behaviour in plain language
- Do NOT use the exact phrase "Is X still affected by Y?"
- End with a question mark

Output only the question, nothing else."""

CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)


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


DISTRACTOR_TEMPLATES = [
    "{cve} has been identified in {product}. The flaw permits {attack} by unauthenticated attackers. "
    "Severity: {sev}. Affected versions: {product} up to {ver}. No patch currently available.",
]

ATTACKS = ["remote code execution", "SQL injection", "buffer overflow",
           "path traversal", "cross-site scripting", "privilege escalation"]
SEVS = ["Critical", "High", "Medium"]
PRODUCTS = ["nginx", "openssl", "redis", "django", "flask", "rails",
            "postgresql", "mysql", "tomcat", "spring", "curl", "libssl"]


def build_corpus(examples: list[dict], n_distractors: int = 1000) -> list[Record]:
    rng = random.Random(42)
    records = []

    # CVE disclosures (anchor documents)
    for ex in examples:
        kb = ex["kb"]
        disc = next((e for e in kb if e["event_type"] == "cve_disclosed"), None)
        if disc:
            records.append(Record(
                record_type="cve_disclosed",
                timestamp=datetime.fromisoformat(disc["timestamp"]),
                text=disc["text"],
                entities={"product": ex["employee"], "cve_id": ex["ticker"]},
            ))

    # Patch releases (gold documents)
    for ex in examples:
        kb = ex["kb"]
        patch = next((e for e in kb if e["event_type"] == "patch_released"), None)
        if patch:
            records.append(Record(
                record_type="patch_released",
                timestamp=datetime.fromisoformat(patch["timestamp"]),
                text=patch["text"],
                entities={"product": ex["employee"], "cve_id": ex["ticker"]},
            ))

    # Distractors (synthetic CVE disclosures with different CVE IDs)
    for i in range(n_distractors):
        product = rng.choice(PRODUCTS)
        attack  = rng.choice(ATTACKS)
        sev     = rng.choice(SEVS)
        cve     = f"CVE-2023-{90000+i:05d}"
        ver     = f"{rng.randint(1,9)}.{rng.randint(0,9)}"
        text = (
            f"{cve} has been identified in {product}. "
            f"The flaw permits {attack} by unauthenticated remote attackers. "
            f"Severity: {sev}. Affected versions: {product} up to {ver}. "
            f"No patch is currently available."
        )
        records.append(Record(
            record_type="distractor",
            timestamp=datetime(2023, 6, 1),
            text=text,
            entities={"product": product, "cve_id": cve},
        ))

    return records


def main():
    lines = open(CVE_DATA).readlines()
    all_examples = [json.loads(l) for l in lines]
    t1_examples = [e for e in all_examples if "type1" in e["hop_type"]]
    print(f"T1 examples: {len(t1_examples)}")

    # Load cache
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["id"]] = entry["query"]
        print(f"Loaded {len(cache)} cached queries")

    # Generate free-form queries
    new_entries = []
    for i, ex in enumerate(t1_examples):
        eid = ex["example_id"]
        if eid in cache:
            continue
        kb = ex["kb"]
        disc = next((e for e in kb if e["event_type"] == "cve_disclosed"), None)
        if not disc:
            continue

        prompt = FREEFORM_PROMPT.format(disclosure_text=disc["text"][:600])
        query = llm_complete(prompt)
        if not query:
            continue

        query = query.strip('"\'').strip()
        if CVE_RE.search(query):
            query = CVE_RE.sub("", query).strip()

        cache[eid] = query
        new_entries.append({"id": eid, "query": query})

        if (i + 1) % 50 == 0 or i < 3:
            print(f"  [{i+1:>3}/{len(t1_examples)}] {query[:70]}")

        time.sleep(0.2)

    if new_entries:
        with open(CACHE_PATH, "a") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")
        print(f"Saved {len(new_entries)} new queries")

    # Build corpus
    corpus = build_corpus(t1_examples)
    n_disc  = sum(1 for r in corpus if r.record_type == "cve_disclosed")
    n_patch = sum(1 for r in corpus if r.record_type == "patch_released")
    disc_start  = 0
    patch_start = n_disc
    print(f"\nCorpus: {len(corpus)} records ({n_disc} disclosures + {n_patch} patches + distractors)")

    # Build pipeline
    adapter = SecurityAdapter()
    pipeline = make_pipeline(corpus, adapter)
    bm25_standalone = _BM25([_tokenize(r.text) for r in corpus])

    k, k1 = 5, 5
    results = []

    for i, ex in enumerate(t1_examples):
        eid = ex["example_id"]
        ff_query = cache.get(eid)
        if not ff_query:
            continue

        gold_idx   = patch_start + i
        anchor_idx = disc_start + i

        bm25_scores = bm25_standalone.scores(ff_query)
        bm25_ranking = sorted(range(len(corpus)), key=lambda j: -bm25_scores[j])
        bm25_rank    = bm25_ranking.index(gold_idx) + 1
        anchor_rank  = bm25_ranking.index(anchor_idx) + 1

        ts_ranking = pipeline.retrieve(ff_query, k1=k1)
        ts_rank    = ts_ranking.index(gold_idx) + 1

        results.append({
            "id": eid, "product": ex["employee"], "cve_id": ex["ticker"],
            "bm25_rank": bm25_rank, "bm25_tca5": int(bm25_rank <= k),
            "anchor_bm25_rank": anchor_rank, "anchor_in_top5": int(anchor_rank <= k),
            "twostage_rank": ts_rank, "twostage_tca5": int(ts_rank <= k),
        })

        if (i + 1) % 50 == 0 or i < 3:
            print(f"  [{i+1:>3}/{len(t1_examples)}] bm25={bm25_rank:>5} ts={ts_rank:>3} "
                  f"anchor@5={'Y' if anchor_rank<=k else 'N'}")

    n = len(results)
    print(f"\n=== CVE Synthetic Two-Stage (n={n}, k={k}) ===")
    bm25_tca = sum(r["bm25_tca5"] for r in results) / n
    ts_tca   = sum(r["twostage_tca5"] for r in results) / n
    anchor_r5 = sum(1 for r in results if r["anchor_bm25_rank"] <= k) / n
    print(f"BM25 TCA@5:      {bm25_tca:.4f}")
    print(f"Two-Stage TCA@5: {ts_tca:.4f}")
    print(f"Stage-1 Anchor Recall@5: {anchor_r5:.4f}")

    agg = {
        "n": n, "k": k, "k1": k1, "corpus_size": len(corpus),
        "bm25_tca5": round(bm25_tca, 4),
        "twostage_tca5": round(ts_tca, 4),
        "stage1_anchor_recall_at_5": round(anchor_r5, 4),
    }
    out = {"aggregate": agg, "per_example": results}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
