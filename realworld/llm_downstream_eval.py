"""
Experiment D: Downstream LLM Answer Quality — Dense RAG vs Two-Stage RAG.

Hypothesis: Dense retrieval silently provides GPT-4o with superseded evidence
(CVE disclosure saying "no patch available"), causing the LLM to confidently
assert the vulnerability is unpatched — even though a patch has existed for months.
Two-stage retrieval provides the correct patch note, enabling correct answers.

Setup:
  - 159 GHSA pairs, free-form queries (existing cache)
  - Dense condition: top-5 from MiniLM dense retrieval
  - Two-stage condition: top-5 from BM25 two-stage pipeline (k1=5)
  - LLM: GPT-4o-mini (cost-efficient, still strong reasoning)
  - Evaluation: answer_correct (does LLM correctly say CVE is patched + right version?)
                confident_wrong (does LLM explicitly say "no patch" / "unpatched"?)

Gold: for all 159 pairs a patch exists — the correct answer is always:
  "Yes, patched. Fixed in version X.Y.Z."

Output:
  data/llm_downstream_results.json
  data/llm_downstream_cache.jsonl   (per-pair LLM responses)
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

ROOT = Path(__file__).parent.parent
PAIRS_PATH    = ROOT / "data" / "ghsa_real_pairs.json"
FREEFORM_PATH = ROOT / "data" / "ghsa_freeform_cache.jsonl"
TS_RESULTS    = ROOT / "data" / "ghsa_twostage_results.json"
CACHE_OUT     = ROOT / "data" / "llm_downstream_cache.jsonl"
RESULTS_OUT   = ROOT / "data" / "llm_downstream_results.json"

MODEL    = "gpt-4o-mini"
MAX_DOC_CHARS = 1200   # truncate each retrieved doc to keep prompts manageable
K = 5

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_pairs() -> list[dict]:
    return json.load(open(PAIRS_PATH))


def load_freeform() -> dict[str, str]:
    d = {}
    for line in open(FREEFORM_PATH):
        if line.strip():
            e = json.loads(line)
            d[e["id"]] = e["query"]
    return d


def load_twostage_results() -> dict[str, dict]:
    """Returns per-pair results keyed by pair id."""
    data = json.load(open(TS_RESULTS))
    return {r["id"]: r for r in data["per_example"]}


# ---------------------------------------------------------------------------
# Build dense rankings (re-use MiniLM from scratch — fast, 2318-doc corpus)
# ---------------------------------------------------------------------------
def build_dense_rankings(pairs: list[dict], freeform: dict[str, str]) -> dict[str, list[int]]:
    """Returns {pair_id: ranked list of doc indices in 2318-doc corpus}"""
    import random
    from sentence_transformers import SentenceTransformer

    # Rebuild corpus (same as ghsa_twostage_eval.py)
    docs, doc_metas = [], []
    disc_start = 0
    for p in pairs:
        docs.append(p["disclosure_text"])
        doc_metas.append({"type": "disclosure", "cve_id": p["cve_id"],
                          "product": p["product"], "pair_idx": len(doc_metas)})
    patch_start = len(docs)
    for i, p in enumerate(pairs):
        docs.append(p["patch_text"])
        doc_metas.append({"type": "patch", "cve_id": p["cve_id"],
                          "product": p["product"], "pair_idx": i})

    rng = random.Random(42)
    products = ["nginx","openssl","log4j","redis","django","flask","rails",
                "postgresql","mysql","tomcat","spring","hibernate","curl","wget",
                "libssl","libjpeg","zlib","expat","glibc","libxml2"]
    severities = ["Critical","High","Medium"]
    attacks = ["remote code execution","SQL injection","buffer overflow",
               "path traversal","cross-site scripting","privilege escalation"]
    for i in range(2000):
        product = rng.choice(products)
        severity = rng.choice(severities)
        attack = rng.choice(attacks)
        cve = f"CVE-2023-{90000+i:05d}"
        ver = f"{rng.randint(1,9)}.{rng.randint(0,9)}"
        docs.append(
            f"{cve} has been identified in {product}. "
            f"The flaw permits {attack} by unauthenticated remote attackers. "
            f"Severity: {severity}. Affected versions: {product} up to and including {ver}. "
            f"No patch is currently available. Exploitation has been observed in the wild."
        )
        doc_metas.append({"type": "distractor", "cve_id": cve, "product": product, "pair_idx": -1})

    print(f"Corpus: {len(docs)} docs")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Encoding corpus with MiniLM...")
    doc_embs = model.encode(docs, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    rankings: dict[str, list[int]] = {}
    for i, pair in enumerate(pairs):
        pid = pair["id"]
        if pid not in freeform:
            continue
        q_emb = model.encode([freeform[pid]], normalize_embeddings=True)
        scores = (q_emb @ doc_embs.T)[0]
        rankings[pid] = list(np.argsort(-scores))

    return rankings, docs, doc_metas, disc_start, patch_start


# ---------------------------------------------------------------------------
# Two-stage retrieve (mirrors ghsa_twostage_eval.py)
# ---------------------------------------------------------------------------
def twostage_retrieve(stage1_ranking: list[int], doc_metas: list[dict],
                      entity_index: dict[tuple, int], k1: int = K) -> list[int]:
    patch_candidates, seen = [], set()
    for idx in stage1_ranking[:k1]:
        m = doc_metas[idx]
        if m["type"] == "disclosure":
            key = (m["product"].lower().strip(), m["cve_id"].upper().strip())
            patch_idx = entity_index.get(key)
            if patch_idx is not None and patch_idx not in seen:
                patch_candidates.append(patch_idx)
                seen.add(patch_idx)
    remaining = [d for d in stage1_ranking if d not in seen]
    return patch_candidates + remaining


def build_entity_index(doc_metas: list[dict], patch_start: int) -> dict[tuple, int]:
    idx = {}
    for i in range(patch_start, len(doc_metas)):
        m = doc_metas[i]
        if m["type"] == "patch":
            idx[(m["product"].lower().strip(), m["cve_id"].upper().strip())] = i
    return idx


def build_bm25_rankings(pairs, freeform, docs, doc_metas):
    from rank_bm25 import BM25Okapi
    tokenized = [re.sub(r"[^\w\s]", " ", d.lower()).split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    rankings = {}
    for pair in pairs:
        pid = pair["id"]
        if pid not in freeform:
            continue
        q_toks = re.sub(r"[^\w\s]", " ", freeform[pid].lower()).split()
        scores = bm25.get_scores(q_toks)
        rankings[pid] = list(np.argsort(-scores))
    return rankings


# ---------------------------------------------------------------------------
# GPT-4o prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a security analyst answering questions about software vulnerabilities.
Answer ONLY based on the provided documents. Be direct and specific.
If the documents indicate a patch exists, state the version that fixes the issue.
If the documents only describe a vulnerability without a fix, say so explicitly."""

def build_user_prompt(query: str, docs_text: list[str]) -> str:
    docs_block = "\n\n---\n\n".join(
        f"[Document {i+1}]\n{d[:MAX_DOC_CHARS]}" for i, d in enumerate(docs_text)
    )
    return (
        f"Question: {query}\n\n"
        f"Retrieved documents:\n\n{docs_block}\n\n"
        f"Based solely on the above documents, answer the question. "
        f"State clearly: (1) Is the vulnerability patched? "
        f"(2) If patched, which version fixes it?"
    )


def call_gpt(query: str, docs_text: list[str]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(query, docs_text)},
        ],
        max_tokens=300,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------
VERSION_PATTERN = re.compile(r"\b(\d+\.\d+(?:\.\d+)*(?:[._-]\w+)?)\b")

PATCHED_SIGNALS = [
    r"\bfixed\b", r"\bpatched\b", r"\bremediated\b", r"\baddressed\b",
    r"\bmitigated\b", r"\bupgrade\b", r"\bupdate\b", r"\brelease[sd]?\b",
    r"\bresolve[sd]?\b", r"version\s+\d", r"\bpatch\s+is\s+available\b",
    r"\bhas been (fixed|patched|resolved|addressed)\b",
]
UNPATCHED_SIGNALS = [
    r"no patch", r"not (yet )?patched", r"unpatched", r"not been patched",
    r"no fix", r"has not been fixed", r"remains (un)?mitigated",
    r"no (known |available )?fix", r"currently (has no|without) (a )?patch",
    r"patch (is )?not (yet )?available",
]


def evaluate_answer(response: str, gold_patch_text: str) -> dict:
    resp_lower = response.lower()

    is_patched_claim  = any(re.search(p, resp_lower) for p in PATCHED_SIGNALS)
    is_unpatched_claim = any(re.search(p, resp_lower) for p in UNPATCHED_SIGNALS)

    # Extract versions mentioned in response and gold
    resp_versions = set(VERSION_PATTERN.findall(response))
    gold_versions = set(VERSION_PATTERN.findall(gold_patch_text))

    version_overlap = bool(resp_versions & gold_versions)

    # Correct = claims patched AND at least one gold version mentioned
    correct = is_patched_claim and not is_unpatched_claim and version_overlap

    # Confident wrong = explicitly says NOT patched (gold says patched)
    confident_wrong = is_unpatched_claim and not is_patched_claim

    return {
        "correct":         correct,
        "confident_wrong": confident_wrong,
        "is_patched_claim":  is_patched_claim,
        "is_unpatched_claim": is_unpatched_claim,
        "version_overlap": version_overlap,
        "resp_versions":   sorted(resp_versions),
        "gold_versions":   sorted(gold_versions),
    }


# ---------------------------------------------------------------------------
# Load cache to resume
# ---------------------------------------------------------------------------
def load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE_OUT.exists():
        for line in CACHE_OUT.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                cache[e["id"]] = e
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairs    = load_pairs()
    freeform = load_freeform()
    print(f"Loaded {len(pairs)} pairs, {len(freeform)} free-form queries")

    # Build corpus + rankings
    dense_rankings, docs, doc_metas, disc_start, patch_start = build_dense_rankings(
        pairs, freeform
    )
    entity_index = build_entity_index(doc_metas, patch_start)

    print("Building BM25 rankings...")
    bm25_rankings = build_bm25_rankings(pairs, freeform, docs, doc_metas)

    cache = load_cache()
    print(f"Cache: {len(cache)} already processed")

    results = []
    total = sum(1 for p in pairs if p["id"] in freeform)

    with open(CACHE_OUT, "a") as cache_f:
        for i, pair in enumerate(pairs):
            pid = pair["id"]
            if pid not in freeform:
                continue

            if pid in cache:
                results.append(cache[pid])
                continue

            query        = freeform[pid]
            gold_patch   = pair["patch_text"]
            gold_patch_idx = patch_start + i

            # Dense top-5 docs
            dense_top5_idx  = dense_rankings[pid][:K]
            dense_top5_docs = [docs[j] for j in dense_top5_idx]

            # Two-stage top-5 docs
            bm25_stage1 = bm25_rankings[pid]
            ts_ranking  = twostage_retrieve(bm25_stage1, doc_metas, entity_index, k1=K)
            ts_top5_idx = ts_ranking[:K]
            ts_top5_docs = [docs[j] for j in ts_top5_idx]

            # Ground truth retrieval
            dense_has_patch  = gold_patch_idx in dense_top5_idx
            ts_has_patch     = gold_patch_idx in ts_top5_idx

            # GPT-4o answers
            try:
                dense_answer = call_gpt(query, dense_top5_docs)
                time.sleep(0.3)
                ts_answer    = call_gpt(query, ts_top5_docs)
                time.sleep(0.3)
            except Exception as e:
                print(f"  [{i+1}] API error for {pid}: {e}")
                time.sleep(5)
                continue

            dense_eval = evaluate_answer(dense_answer, gold_patch)
            ts_eval    = evaluate_answer(ts_answer,    gold_patch)

            row = {
                "id":              pid,
                "cve_id":          pair["cve_id"],
                "product":         pair["product"],
                "query":           query,
                "dense_has_patch_in_top5":    dense_has_patch,
                "ts_has_patch_in_top5":       ts_has_patch,
                "dense_answer":    dense_answer,
                "ts_answer":       ts_answer,
                "dense_eval":      dense_eval,
                "ts_eval":         ts_eval,
            }
            results.append(row)
            cache_f.write(json.dumps(row) + "\n")
            cache_f.flush()

            d_sym = "✓" if dense_eval["correct"] else ("✗!" if dense_eval["confident_wrong"] else "?")
            t_sym = "✓" if ts_eval["correct"] else ("✗!" if ts_eval["confident_wrong"] else "?")
            n_done = len(results)
            print(f"  [{n_done:>3}/{total}] {pid:12s}  dense={d_sym}  ts={t_sym}  "
                  f"(patch_in_dense={dense_has_patch}, patch_in_ts={ts_has_patch})")

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    n = len(results)
    dense_correct       = sum(r["dense_eval"]["correct"]         for r in results)
    dense_conf_wrong    = sum(r["dense_eval"]["confident_wrong"] for r in results)
    ts_correct          = sum(r["ts_eval"]["correct"]            for r in results)
    ts_conf_wrong       = sum(r["ts_eval"]["confident_wrong"]    for r in results)
    dense_has_patch     = sum(r["dense_has_patch_in_top5"]       for r in results)
    ts_has_patch        = sum(r["ts_has_patch_in_top5"]          for r in results)

    agg = {
        "n": n,
        "model": MODEL,
        "k": K,
        "retrieval": {
            "dense_patch_recall5":     round(dense_has_patch / n, 3),
            "twostage_patch_recall5":  round(ts_has_patch    / n, 3),
        },
        "answer_accuracy": {
            "dense_correct":           round(dense_correct    / n, 3),
            "twostage_correct":        round(ts_correct       / n, 3),
            "delta_pp":                round((ts_correct - dense_correct) / n * 100, 1),
        },
        "confident_wrong": {
            "dense_confident_wrong":   round(dense_conf_wrong / n, 3),
            "twostage_confident_wrong":round(ts_conf_wrong    / n, 3),
            "dense_conf_wrong_n":      dense_conf_wrong,
            "twostage_conf_wrong_n":   ts_conf_wrong,
        },
    }

    print(f"\n=== LLM Downstream Results (n={n}, model={MODEL}) ===")
    print(f"  Retrieval:    Dense patch@5={agg['retrieval']['dense_patch_recall5']:.3f}"
          f"  TS patch@5={agg['retrieval']['twostage_patch_recall5']:.3f}")
    print(f"  Correct:      Dense={agg['answer_accuracy']['dense_correct']:.3f}"
          f"  TS={agg['answer_accuracy']['twostage_correct']:.3f}"
          f"  Δ={agg['answer_accuracy']['delta_pp']:+.1f}pp")
    print(f"  Conf.wrong:   Dense={dense_conf_wrong}/{n}"
          f"  TS={ts_conf_wrong}/{n}")

    out = {"aggregate": agg, "per_example": results}
    with open(RESULTS_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {RESULTS_OUT}")


if __name__ == "__main__":
    main()
