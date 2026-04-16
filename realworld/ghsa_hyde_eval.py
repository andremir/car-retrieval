"""
HyDE (Hypothetical Document Embeddings) evaluation on 159 real GHSA advisories.

For each CVE query, we:
  1. Prompt Llama 3.3 70B to generate a hypothetical patch release note
  2. Embed the hypothetical with MiniLM (same as main eval)
  3. Retrieve by cosine similarity against the 2318-document corpus
  4. Report TCA@k

Key prediction (Theorem 1):
  The LLM will tend to include the CVE identifier in its hypothetical
  (the query explicitly mentions it), so the hypothetical embeds closer
  to CVE disclosures than to real patch notes — perpetuating the gap.
  Expected TCA@5 < entity probe (1.000) and possibly close to naive dense.

Output: data/ghsa_hyde_results.json
"""
from __future__ import annotations
import os
import json, math, re, random, time, os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import urllib.request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_URL = "https://api.studio.nebius.ai/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

PAIRS_PATH = Path(__file__).parent.parent / "data" / "ghsa_real_pairs.json"
OUT_PATH   = Path(__file__).parent.parent / "data" / "ghsa_hyde_results.json"

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def llm_complete(prompt: str, max_tokens: int = 300, retries: int = 3) -> str | None:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode()
    headers = {
        "Authorization": f"Bearer {NEBIUS_API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(NEBIUS_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    LLM error: {e}")
                return None


# ---------------------------------------------------------------------------
# HyDE prompt
# ---------------------------------------------------------------------------
HYDE_PROMPT = """\
A developer asks: "{query}"

Write a plausible GitHub release note that would be the patch for this vulnerability.
Write only the release note body — no preamble, no meta-commentary.
The release note should sound like a real software release: version number, what changed, why.
Keep it under 150 words."""


def generate_hypothetical(product: str, cve_id: str, query: str) -> str | None:
    prompt = HYDE_PROMPT.format(query=query)
    return llm_complete(prompt, max_tokens=250)


# ---------------------------------------------------------------------------
# Build corpus (identical to main eval)
# ---------------------------------------------------------------------------
def build_corpus(pairs: list[dict], n_distractors: int = 2000):
    docs, doc_metas = [], []
    for p in pairs:
        docs.append(p["disclosure_text"])
        doc_metas.append({"type": "disclosure", "product": p["product"], "cve_id": p["cve_id"]})
    patch_start = len(docs)
    for p in pairs:
        docs.append(p["patch_text"])
        doc_metas.append({"type": "patch", "product": p["product"], "cve_id": p["cve_id"]})

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
        doc_metas.append({"type": "distractor", "product": product, "cve_id": cve})

    return docs, doc_metas, patch_start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairs = json.load(open(PAIRS_PATH))
    print(f"Loaded {len(pairs)} pairs")

    docs, doc_metas, patch_start = build_corpus(pairs)
    print(f"Corpus: {len(docs)} docs")

    # Pre-compute document embeddings once
    print("Loading MiniLM and encoding docs...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    d_embs = model.encode(docs, normalize_embeddings=True,
                          show_progress_bar=False, batch_size=64)
    print(f"Doc embeddings: {d_embs.shape}")

    # Check for cached hypotheticals
    cache_path = Path(__file__).parent.parent / "data" / "ghsa_hyde_cache.jsonl"
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["id"]] = entry["hypothetical"]
        print(f"Loaded {len(cache)} cached hypotheticals")

    k = 5
    hyde_results = []
    new_cache_entries = []

    for i, pair in enumerate(pairs):
        product, cve_id = pair["product"], pair["cve_id"]
        query = f"Is {product} still affected by {cve_id}? Has this security flaw been addressed and is the system still at risk?"
        gold_idx = patch_start + i

        # Get or generate hypothetical
        if pair["id"] in cache:
            hyp = cache[pair["id"]]
        else:
            hyp = generate_hypothetical(product, cve_id, query)
            if hyp:
                new_cache_entries.append({"id": pair["id"], "hypothetical": hyp})
                cache[pair["id"]] = hyp
            time.sleep(0.3)

        if not hyp:
            print(f"  [{i+1}/{len(pairs)}] SKIP (no hypothetical)")
            continue

        # Embed hypothetical and rank
        hyp_emb = model.encode([hyp], normalize_embeddings=True)
        scores = (hyp_emb @ d_embs.T)[0]
        ranking = list(np.argsort(-scores))
        rank = ranking.index(gold_idx) + 1
        tca = int(rank <= k)

        # Also rank with original query (for comparison)
        q_emb = model.encode([query], normalize_embeddings=True)
        q_scores = (q_emb @ d_embs.T)[0]
        q_ranking = list(np.argsort(-q_scores))
        q_rank = q_ranking.index(gold_idx) + 1

        # Check if hypothetical contains CVE ID (key diagnostic)
        hyp_has_cve = cve_id.lower() in hyp.lower()

        hyde_results.append({
            "id": pair["id"], "cve_id": cve_id, "product": product,
            "hyde_rank": rank, "hyde_tca": tca,
            "query_rank": q_rank,  # baseline dense rank
            "hypothetical_has_cve_id": hyp_has_cve,
            "hypothetical_snippet": hyp[:200],
        })

        if (i + 1) % 10 == 0 or i < 5:
            cve_flag = "CVE" if hyp_has_cve else "   "
            print(f"  [{i+1:>3}/{len(pairs)}] hyde_rank={rank:>5} q_rank={q_rank:>5} "
                  f"tca={tca} {cve_flag} | {product[:20]}")

    # Append new cache entries
    if new_cache_entries:
        with open(cache_path, "a") as f:
            for entry in new_cache_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {len(new_cache_entries)} new cache entries")

    if not hyde_results:
        print("ERROR: no results")
        return

    n = len(hyde_results)
    hyde_tcas = [r["hyde_tca"] for r in hyde_results]
    hyde_ranks = [r["hyde_rank"] for r in hyde_results]
    n_cve_in_hyp = sum(1 for r in hyde_results if r["hypothetical_has_cve_id"])

    print(f"\n=== HyDE Results (k={k}, n={n}) ===")
    print(f"TCA@5:     {sum(hyde_tcas)/n:.4f} ({sum(hyde_tcas)}/{n})")
    print(f"Mean rank: {sum(hyde_ranks)/n:.1f}")
    print(f"Median:    {sorted(hyde_ranks)[n//2]}")
    print(f"CVE ID in hypothetical: {n_cve_in_hyp}/{n} = {n_cve_in_hyp/n*100:.0f}%")

    # TCA@10, @20
    for kk in [10, 20]:
        tca_k = sum(1 for r in hyde_ranks if r <= kk) / n
        print(f"TCA@{kk}:    {tca_k:.4f}")

    # Split: hypotheticals WITH vs WITHOUT CVE ID
    with_cve = [r for r in hyde_results if r["hypothetical_has_cve_id"]]
    without_cve = [r for r in hyde_results if not r["hypothetical_has_cve_id"]]
    if with_cve:
        tca_with = sum(r["hyde_tca"] for r in with_cve) / len(with_cve)
        print(f"\nHyDE when hypothetical CONTAINS CVE ID (n={len(with_cve)}): TCA@5={tca_with:.4f}")
    if without_cve:
        tca_without = sum(r["hyde_tca"] for r in without_cve) / len(without_cve)
        print(f"HyDE when hypothetical OMITS CVE ID    (n={len(without_cve)}): TCA@5={tca_without:.4f}")

    results = {
        "aggregate": {
            "n": n,
            "tca5": round(sum(hyde_tcas)/n, 4),
            "tca10": round(sum(1 for r in hyde_ranks if r <= 10)/n, 4),
            "tca20": round(sum(1 for r in hyde_ranks if r <= 20)/n, 4),
            "mean_rank": round(sum(hyde_ranks)/n, 1),
            "median_rank": sorted(hyde_ranks)[n//2],
            "corpus_size": len(docs),
            "k": k,
            "pct_hypothetical_contains_cve_id": round(n_cve_in_hyp/n*100, 1),
        },
        "split_by_cve_in_hyp": {
            "with_cve": {
                "n": len(with_cve),
                "tca5": round(sum(r["hyde_tca"] for r in with_cve)/max(len(with_cve),1), 4),
            } if with_cve else None,
            "without_cve": {
                "n": len(without_cve),
                "tca5": round(sum(r["hyde_tca"] for r in without_cve)/max(len(without_cve),1), 4),
            } if without_cve else None,
        },
        "per_example": hyde_results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
