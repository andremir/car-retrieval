"""
Free-form query generation via Llama 3.3 70B (OpenAI-compatible API).

For each GHSA pair, generate ONE natural-language question that:
  - Describes the vulnerability in natural language
  - Does NOT contain the CVE identifier string
  - Does NOT use the structured template "Is X still affected by Y?"

Writes: data/ghsa_freeform_cache.jsonl
Each line: {"id": str, "query": str}

Usage: python3 ghsa_freeform_gen.py
"""
from __future__ import annotations
import os
import json, re, time, urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "")
NEBIUS_URL = "https://api.studio.nebius.ai/v1/chat/completions"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

PAIRS_PATH = Path(__file__).parent.parent / "data" / "ghsa_real_pairs.json"
CACHE_PATH = Path(__file__).parent.parent / "data" / "ghsa_freeform_cache.jsonl"

CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)

PROMPT_TEMPLATE = """\
You are a software developer who read this security advisory:

{disclosure}

Write ONE natural English question (maximum 50 words) that a developer would ask to find out whether this vulnerability has been patched in the software.

Rules:
- Do NOT include any CVE identifier (e.g. CVE-2024-XXXXX)
- Do NOT use the exact phrase "Is X still affected by Y?"
- Describe the specific vulnerability behaviour in plain language
- End with a question mark

Output only the question, nothing else."""


def llm_complete(prompt: str, retries: int = 3) -> str | None:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 120,
        "temperature": 0.2,
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


def main():
    pairs = json.load(open(PAIRS_PATH))
    print(f"Loaded {len(pairs)} pairs")

    # Load cache
    cache: dict[str, str] = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                cache[entry["id"]] = entry["query"]
        print(f"Loaded {len(cache)} cached queries")

    new_entries = []
    failed = []

    for i, pair in enumerate(pairs):
        pid = pair["id"]
        if pid in cache:
            continue

        disclosure = pair["disclosure_text"][:800]
        prompt = PROMPT_TEMPLATE.format(disclosure=disclosure)

        query = llm_complete(prompt)

        if not query:
            print(f"  [{i+1:>3}/{len(pairs)}] FAILED: {pid}")
            failed.append(pid)
            continue

        # Strip quotes if LLM wrapped them
        query = query.strip('"').strip("'").strip()

        # Verify no CVE ID
        if CVE_RE.search(query):
            # Try to strip it
            query_clean = CVE_RE.sub("", query).strip(" ,.")
            print(f"  [{i+1:>3}/{len(pairs)}] CVE stripped: {pair['cve_id']} | {query_clean[:60]}")
            query = query_clean

        cache[pid] = query
        new_entries.append({"id": pid, "query": query})

        if (i + 1) % 20 == 0 or i < 3:
            print(f"  [{i+1:>3}/{len(pairs)}] {query[:80]}")

        time.sleep(0.25)

    # Save new entries
    if new_entries:
        with open(CACHE_PATH, "a") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")
        print(f"\nSaved {len(new_entries)} new queries → {CACHE_PATH}")

    if failed:
        print(f"Failed: {len(failed)} pairs")

    # Summary
    total = len(cache)
    n_with_cve = sum(1 for q in cache.values() if CVE_RE.search(q))
    print(f"\nTotal cached: {total}/{len(pairs)}")
    print(f"Queries still containing CVE ID: {n_with_cve}")

    # Show samples
    print("\nSample queries:")
    for pid, q in list(cache.items())[:5]:
        print(f"  {pid}: {q}")


if __name__ == "__main__":
    main()
