"""
vocab_gap_analysis.py

Measures the vocabulary gap between:
  (A) user queries about vulnerabilities
  (B) CVE / GHSA disclosure texts
  (C) simulated patch/release-note texts

The core claim: patch announcements use systematically different vocabulary
than CVE disclosures and the user queries that reference them.

Two data sources:
  1. GitHub Security Advisories API (GHSA) — no auth needed for public data
  2. NVD CVE API — public, no key needed for 5 req/30s
"""

import requests
import json
import re
import math
import time
from collections import Counter

# ── helpers ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, keep words ≥ 2 chars."""
    return re.findall(r"\b[a-z]{2,}\b", text.lower())

def token_set(text: str) -> set[str]:
    return set(tokenize(text))

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# ── BM25 (single-query, multi-doc corpus) ────────────────────────────────────

class BM25:
    """Okapi BM25 with k1=1.5, b=0.75."""
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        # term → document-frequency
        self.df: dict[str, int] = Counter()
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1
        # pre-compute idf
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens: list[str], doc_index: int) -> float:
        doc = self.corpus[doc_index]
        dl = len(doc)
        tf_map: dict[str, int] = Counter(doc)
        s = 0.0
        for term in set(query_tokens):
            if term not in self.idf:
                continue
            tf = tf_map.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf[term] * numerator / denominator
        return s

# ── vocabulary diversity / lexical uniqueness ────────────────────────────────

STOPWORDS = {
    "the", "a", "an", "is", "in", "it", "of", "to", "and", "or", "for",
    "on", "at", "by", "with", "from", "that", "this", "are", "be", "was",
    "has", "have", "had", "not", "but", "as", "if", "its", "via", "can",
    "could", "will", "would", "may", "should", "still", "been", "when",
    "which", "also", "into", "than", "up", "more", "their", "they", "we",
    "you", "he", "she", "do", "did", "so", "then", "any", "all",
}

def content_words(text: str) -> set[str]:
    return token_set(text) - STOPWORDS


# ── GHSA data source ─────────────────────────────────────────────────────────

def fetch_ghsa(n: int = 60) -> list[dict]:
    """Fetch up to n GitHub security advisories (HIGH severity, reviewed)."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    per_page = min(n, 100)
    url = "https://api.github.com/advisories"
    params = {"type": "reviewed", "per_page": per_page, "severity": "high"}
    print(f"Fetching {per_page} GHSA advisories…")
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Received {len(data)} advisories")
    return data


# ── NVD data source ──────────────────────────────────────────────────────────

def fetch_nvd(n: int = 60) -> list[dict]:
    """
    Fetch up to n recent HIGH-severity CVEs from the NVD 2.0 API.
    Strategy: probe the total count first, then fetch from near the end
    of the result list (most recently published) using cvssV3Severity=HIGH.
    """
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

    # First: get the total count
    probe = requests.get(
        url,
        params={"resultsPerPage": 1, "startIndex": 0, "cvssV3Severity": "HIGH"},
        timeout=30,
    )
    probe.raise_for_status()
    total = probe.json().get("totalResults", 0)
    print(f"  NVD total HIGH-severity CVEs: {total}")

    # Fetch from near the end to get the most recently published
    start_idx = max(0, total - n)
    params = {
        "resultsPerPage": min(n, 2000),
        "startIndex": start_idx,
        "cvssV3Severity": "HIGH",
    }
    print(f"  Fetching {n} NVD CVEs starting at index {start_idx}…")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("vulnerabilities", [])
    print(f"  Received {len(items)} CVEs")
    return items


# ── patch-text simulator ─────────────────────────────────────────────────────

# Real vendor release notes / changelogs tend to use:
#   - version numbers ("3.2.1", "v4.0")
#   - generic verbs ("update", "upgrade", "fixed", "resolved", "patch", "release")
#   - component-centric language ("the dependency", "the library", "the module")
#   - absence of: CVE IDs, CVSS scores, CWE IDs, attacker-framing
#
# We simulate this by:
#   (a) stripping identifiers (CVE-*, GHSA-*, CWE-*, CVSS: …) from the disclosure
#   (b) replacing attacker-framing keywords with patch-framing equivalents
#   (c) prepending a realistic release-note header

ATTACKER_TO_PATCH = {
    r"\battacker[s]?\b": "the maintainers",
    r"\bexploit[s]?\b": "address",
    r"\bvulnerabilit[y|ies]+\b": "issue",
    r"\bmalicious\b": "resolved",
    r"\breveal[s]?\b": "disclose",
    r"\bexpos[es]+\b": "resolve",
    r"\bbypass[es]+\b": "handle",
    r"\binjection\b": "input handling",
    r"\bexecution of arbitrary code\b": "a code execution issue",
    r"\bunauthorized\b": "unintended",
    r"\bremote code execution\b": "a code execution issue",
}

ID_PATTERNS = [
    r"CVE-\d{4}-\d+",
    r"GHSA-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}-[a-zA-Z0-9]{4}",
    r"CWE-\d+",
    r"CVSS[^\s]*\s*[\d.]+",
]


def simulate_patch_text(
    disclosure: str,
    cve_id: str = "",
    ghsa_id: str = "",
    packages: list[str] = None,
    version: str = "3.x",
) -> str:
    """
    Produce a realistic release-note-style text from a CVE description by:
    1. removing security identifiers (CVE IDs, CVSS, CWE)
    2. softening attacker-framing language
    3. replacing package names with generic references
    4. prepending a release-note header
    """
    text = disclosure

    # Strip identifiers
    for pat in ID_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Soften attacker framing
    for pat, replacement in ATTACKER_TO_PATCH.items():
        text = re.sub(pat, replacement, text, flags=re.IGNORECASE)

    # Replace specific package/product names with generic terms
    if packages:
        for pkg in sorted(packages, key=len, reverse=True):  # longest first
            if len(pkg) > 2:
                text = re.sub(re.escape(pkg), "the component", text, flags=re.IGNORECASE)

    # Strip version refs like "before 1.2.3" or "versions < 1.0"
    text = re.sub(r"\b(before|prior to|versions?)\s+[\d.x]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Prepend realistic release-note header
    header = (
        f"Release notes — {version}\n"
        "This release includes maintenance updates and stability improvements. "
        "Users are encouraged to upgrade to the latest version. "
        "We thank the community for responsible reporting. "
    )
    return header + text


# ── process GHSA entries ──────────────────────────────────────────────────────

def process_ghsa(advisories: list[dict]) -> list[dict]:
    results = []
    for adv in advisories:
        desc = (adv.get("description") or "").strip()
        if len(desc) < 40:
            continue

        ghsa_id = adv.get("ghsa_id", "GHSA-UNKNOWN")
        cve_id = adv.get("cve_id") or ghsa_id

        # Collect package names
        packages = []
        for vuln in adv.get("vulnerabilities", []):
            pkg = (vuln.get("package") or {}).get("name", "")
            if pkg:
                packages.append(pkg)
        package_str = packages[0] if packages else "the package"

        # ── three texts ──
        disclosure = desc
        query = (
            f"Is {package_str} still affected by {cve_id}? "
            f"Has this security vulnerability been patched and is the system still at risk?"
        )
        patch = simulate_patch_text(
            disclosure, cve_id=cve_id, ghsa_id=ghsa_id, packages=packages
        )

        # ── token sets ──
        q_set = token_set(query)
        d_set = token_set(disclosure)
        p_set = token_set(patch)

        # content-word sets (no stopwords) — more discriminating
        q_cw = content_words(query)
        d_cw = content_words(disclosure)
        p_cw = content_words(patch)

        # ── Jaccard ──
        j_qd = jaccard(q_set, d_set)
        j_qp = jaccard(q_set, p_set)
        j_dp = jaccard(d_set, p_set)

        j_qd_cw = jaccard(q_cw, d_cw)
        j_qp_cw = jaccard(q_cw, p_cw)

        results.append(
            {
                "source": "GHSA",
                "id": ghsa_id,
                "cve_id": cve_id,
                "package": package_str,
                "disclosure_len": len(disclosure.split()),
                "query_len": len(query.split()),
                "patch_len": len(patch.split()),
                "jaccard_query_disclosure": round(j_qd, 4),
                "jaccard_query_patch": round(j_qp, 4),
                "jaccard_disclosure_patch": round(j_dp, 4),
                "gap_qd_minus_qp": round(j_qd - j_qp, 4),
                "jaccard_cw_query_disclosure": round(j_qd_cw, 4),
                "jaccard_cw_query_patch": round(j_qp_cw, 4),
                "gap_cw": round(j_qd_cw - j_qp_cw, 4),
            }
        )
    return results


# ── process NVD entries ───────────────────────────────────────────────────────

def process_nvd(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        cve = item.get("cve", {})
        cve_id = cve.get("id", "CVE-UNKNOWN")

        # Description
        descs = cve.get("descriptions", [])
        eng = next((d["value"] for d in descs if d.get("lang") == "en"), None)
        if not eng or len(eng) < 40:
            continue

        # Try to extract a product name from CPE or configurations
        packages = []
        for cfg in cve.get("configurations", []):
            for node in cfg.get("nodes", []):
                for cpe_match in node.get("cpeMatch", []):
                    cpe = cpe_match.get("criteria", "")
                    parts = cpe.split(":")
                    if len(parts) >= 5:
                        product = parts[4].replace("_", " ")
                        if product and product != "*":
                            packages.append(product)
                            break
                if packages:
                    break
            if packages:
                break

        # Fallback: extract product-like noun from description
        if not packages:
            # Take the first capitalized multi-word sequence before "in" or "before"
            m = re.search(r"in ([A-Z][A-Za-z0-9 _-]{2,30}) (?:before|through|prior)", eng)
            if m:
                packages.append(m.group(1).strip())

        package_str = packages[0] if packages else "the product"

        disclosure = eng
        query = (
            f"Is {package_str} still affected by {cve_id}? "
            f"Has this security vulnerability been patched and is the system still at risk?"
        )
        patch = simulate_patch_text(
            disclosure, cve_id=cve_id, packages=packages
        )

        q_set = token_set(query)
        d_set = token_set(disclosure)
        p_set = token_set(patch)

        q_cw = content_words(query)
        d_cw = content_words(disclosure)
        p_cw = content_words(patch)

        j_qd = jaccard(q_set, d_set)
        j_qp = jaccard(q_set, p_set)
        j_dp = jaccard(d_set, p_set)
        j_qd_cw = jaccard(q_cw, d_cw)
        j_qp_cw = jaccard(q_cw, p_cw)

        results.append(
            {
                "source": "NVD",
                "id": cve_id,
                "cve_id": cve_id,
                "package": package_str,
                "disclosure_len": len(disclosure.split()),
                "query_len": len(query.split()),
                "patch_len": len(patch.split()),
                "jaccard_query_disclosure": round(j_qd, 4),
                "jaccard_query_patch": round(j_qp, 4),
                "jaccard_disclosure_patch": round(j_dp, 4),
                "gap_qd_minus_qp": round(j_qd - j_qp, 4),
                "jaccard_cw_query_disclosure": round(j_qd_cw, 4),
                "jaccard_cw_query_patch": round(j_qp_cw, 4),
                "gap_cw": round(j_qd_cw - j_qp_cw, 4),
            }
        )
    return results


# ── BM25 analysis ─────────────────────────────────────────────────────────────

def run_bm25_analysis(all_results: list[dict], raw_texts: list[dict]) -> dict:
    """
    Build a BM25 corpus of [disclosure_text, patch_text] per entry,
    then score each query against both.
    Returns mean scores and the ratio.
    """
    disclosure_corpus = [tokenize(r["disclosure_text"]) for r in raw_texts]
    patch_corpus = [tokenize(r["patch_text"]) for r in raw_texts]
    queries = [tokenize(r["query"]) for r in raw_texts]

    bm25_disc = BM25(disclosure_corpus)
    bm25_patch = BM25(patch_corpus)

    bm25_scores = []
    for i, q in enumerate(queries):
        s_disc = bm25_disc.score(q, i)
        s_patch = bm25_patch.score(q, i)
        bm25_scores.append({
            "id": raw_texts[i]["id"],
            "bm25_query_disclosure": round(s_disc, 4),
            "bm25_query_patch": round(s_patch, 4),
            "bm25_gap": round(s_disc - s_patch, 4),
        })

    mean_disc = sum(r["bm25_query_disclosure"] for r in bm25_scores) / len(bm25_scores)
    mean_patch = sum(r["bm25_query_patch"] for r in bm25_scores) / len(bm25_scores)
    ratio = mean_disc / mean_patch if mean_patch > 0 else float("inf")

    return {
        "per_entry": bm25_scores,
        "mean_bm25_query_disclosure": round(mean_disc, 4),
        "mean_bm25_query_patch": round(mean_patch, 4),
        "bm25_ratio_disc_over_patch": round(ratio, 3),
    }


# ── summary statistics ────────────────────────────────────────────────────────

def summarize(results: list[dict], label: str) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def mean(key):
        return round(sum(r[key] for r in results) / n, 4)

    def pct_positive(key):
        return round(sum(1 for r in results if r[key] > 0) / n * 100, 1)

    m_jqd = mean("jaccard_query_disclosure")
    m_jqp = mean("jaccard_query_patch")
    m_gap = mean("gap_qd_minus_qp")
    m_jqd_cw = mean("jaccard_cw_query_disclosure")
    m_jqp_cw = mean("jaccard_cw_query_patch")
    m_gap_cw = mean("gap_cw")

    print(f"\n{'='*60}")
    print(f"  {label}  (n={n})")
    print(f"{'='*60}")
    print(f"  Jaccard(query, disclosure)          = {m_jqd:.4f}")
    print(f"  Jaccard(query, patch/release note)  = {m_jqp:.4f}")
    print(f"  Gap (disclosure − patch)            = {m_gap:+.4f}  ({pct_positive('gap_qd_minus_qp')}% positive)")
    print()
    print(f"  Content-word Jaccard(Q, disclosure) = {m_jqd_cw:.4f}")
    print(f"  Content-word Jaccard(Q, patch)      = {m_jqp_cw:.4f}")
    print(f"  Content-word gap                    = {m_gap_cw:+.4f}  ({pct_positive('gap_cw')}% positive)")

    return {
        "n": n,
        "mean_jaccard_query_disclosure": m_jqd,
        "mean_jaccard_query_patch": m_jqp,
        "mean_gap": m_gap,
        "pct_gap_positive": pct_positive("gap_qd_minus_qp"),
        "mean_cw_jaccard_query_disclosure": m_jqd_cw,
        "mean_cw_jaccard_query_patch": m_jqp_cw,
        "mean_cw_gap": m_gap_cw,
        "pct_cw_gap_positive": pct_positive("gap_cw"),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out_path = "./data/real_cve_vocab_gap.json"

    all_results = []
    raw_texts = []  # for BM25

    # ── GHSA ──
    try:
        ghsa_raw = fetch_ghsa(n=60)
        ghsa_results = process_ghsa(ghsa_raw)
        print(f"  Processed {len(ghsa_results)} GHSA entries")
        all_results.extend(ghsa_results)

        # rebuild raw texts for BM25
        for adv in ghsa_raw:
            desc = (adv.get("description") or "").strip()
            if len(desc) < 40:
                continue
            ghsa_id = adv.get("ghsa_id", "GHSA-UNKNOWN")
            cve_id = adv.get("cve_id") or ghsa_id
            packages = []
            for vuln in adv.get("vulnerabilities", []):
                pkg = (vuln.get("package") or {}).get("name", "")
                if pkg:
                    packages.append(pkg)
            package_str = packages[0] if packages else "the package"
            query = (
                f"Is {package_str} still affected by {cve_id}? "
                f"Has this security vulnerability been patched and is the system still at risk?"
            )
            patch = simulate_patch_text(desc, cve_id=cve_id, ghsa_id=ghsa_id, packages=packages)
            raw_texts.append({"id": ghsa_id, "query": query, "disclosure_text": desc, "patch_text": patch})
    except Exception as e:
        print(f"  GHSA fetch failed: {e}")

    # ── NVD ──
    try:
        time.sleep(1)  # be polite
        nvd_raw = fetch_nvd(n=60)
        nvd_results = process_nvd(nvd_raw)
        print(f"  Processed {len(nvd_results)} NVD entries")
        all_results.extend(nvd_results)

        for item in nvd_raw:
            cve = item.get("cve", {})
            cve_id = cve.get("id", "CVE-UNKNOWN")
            descs = cve.get("descriptions", [])
            eng = next((d["value"] for d in descs if d.get("lang") == "en"), None)
            if not eng or len(eng) < 40:
                continue
            packages = []
            for cfg in cve.get("configurations", []):
                for node in cfg.get("nodes", []):
                    for cpe_match in node.get("cpeMatch", []):
                        cpe = cpe_match.get("criteria", "")
                        parts = cpe.split(":")
                        if len(parts) >= 5:
                            product = parts[4].replace("_", " ")
                            if product and product != "*":
                                packages.append(product)
                                break
                    if packages:
                        break
                if packages:
                    break
            if not packages:
                m = re.search(r"in ([A-Z][A-Za-z0-9 _-]{2,30}) (?:before|through|prior)", eng)
                if m:
                    packages.append(m.group(1).strip())
            package_str = packages[0] if packages else "the product"
            query = (
                f"Is {package_str} still affected by {cve_id}? "
                f"Has this security vulnerability been patched and is the system still at risk?"
            )
            patch = simulate_patch_text(eng, cve_id=cve_id, packages=packages)
            raw_texts.append({"id": cve_id, "query": query, "disclosure_text": eng, "patch_text": patch})
    except Exception as e:
        print(f"  NVD fetch failed: {e}")

    if not all_results:
        print("ERROR: No data fetched. Exiting.")
        return

    # ── summaries ──
    ghsa_res = [r for r in all_results if r["source"] == "GHSA"]
    nvd_res = [r for r in all_results if r["source"] == "NVD"]
    combined_summary = summarize(all_results, "COMBINED (GHSA + NVD)")
    ghsa_summary = summarize(ghsa_res, "GHSA only")
    nvd_summary = summarize(nvd_res, "NVD only")

    # ── BM25 analysis ──
    print(f"\n{'='*60}")
    print("  BM25 analysis (query vs disclosure vs patch)")
    print(f"{'='*60}")
    bm25_out = {}
    if raw_texts:
        bm25_out = run_bm25_analysis(all_results, raw_texts)
        print(f"  Mean BM25(query, disclosure) = {bm25_out['mean_bm25_query_disclosure']:.4f}")
        print(f"  Mean BM25(query, patch)      = {bm25_out['mean_bm25_query_patch']:.4f}")
        print(f"  Ratio (disc/patch)           = {bm25_out['bm25_ratio_disc_over_patch']:.3f}x")

    # ── unique vocabulary sets (macro-level) ──
    print(f"\n{'='*60}")
    print("  Macro vocabulary analysis")
    print(f"{'='*60}")
    all_q_words: set[str] = set()
    all_d_words: set[str] = set()
    all_p_words: set[str] = set()
    for r in raw_texts:
        all_q_words |= content_words(r["query"])
        all_d_words |= content_words(r["disclosure_text"])
        all_p_words |= content_words(r["patch_text"])

    q_only = all_q_words - all_d_words - all_p_words
    d_only = all_d_words - all_q_words - all_p_words
    p_only = all_p_words - all_q_words - all_d_words
    shared_qd = (all_q_words & all_d_words) - all_p_words
    shared_qp = (all_q_words & all_p_words) - all_d_words
    shared_dp = (all_d_words & all_p_words) - all_q_words
    all_shared = all_q_words & all_d_words & all_p_words

    print(f"  Unique vocab — queries:       {len(all_q_words)}")
    print(f"  Unique vocab — disclosures:   {len(all_d_words)}")
    print(f"  Unique vocab — patch notes:   {len(all_p_words)}")
    print(f"  Words only in disclosures:    {len(d_only)}")
    print(f"  Words only in patch notes:    {len(p_only)}")
    print(f"  Shared Q∩D (not P):           {len(shared_qd)}")
    print(f"  Shared Q∩P (not D):           {len(shared_qp)}")
    print(f"  All three shared:             {len(all_shared)}")
    print(f"\n  Sample 'disclosure-only' terms: {sorted(d_only)[:20]}")
    print(f"  Sample 'patch-only' terms:      {sorted(p_only)[:20]}")

    macro_vocab = {
        "query_vocab_size": len(all_q_words),
        "disclosure_vocab_size": len(all_d_words),
        "patch_vocab_size": len(all_p_words),
        "disclosure_only_size": len(d_only),
        "patch_only_size": len(p_only),
        "shared_qd_not_p": len(shared_qd),
        "shared_qp_not_d": len(shared_qp),
        "all_three_shared": len(all_shared),
        "sample_disclosure_only": sorted(d_only)[:30],
        "sample_patch_only": sorted(p_only)[:30],
    }

    # ── save ──
    output = {
        "summary": {
            "combined": combined_summary,
            "ghsa": ghsa_summary,
            "nvd": nvd_summary,
        },
        "bm25": bm25_out,
        "macro_vocab": macro_vocab,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ── final paper-ready numbers ──
    print(f"\n{'='*60}")
    print("  PAPER-READY NUMBERS")
    print(f"{'='*60}")
    cs = combined_summary
    print(f"  n = {cs['n']}")
    print(f"  Jaccard(Q, Disclosure) = {cs['mean_jaccard_query_disclosure']:.4f}")
    print(f"  Jaccard(Q, Patch)      = {cs['mean_jaccard_query_patch']:.4f}")
    print(f"  Absolute gap           = {cs['mean_gap']:+.4f}")
    print(f"  % with positive gap    = {cs['pct_gap_positive']}%")
    print(f"  CW Jaccard(Q, D)       = {cs['mean_cw_jaccard_query_disclosure']:.4f}")
    print(f"  CW Jaccard(Q, P)       = {cs['mean_cw_jaccard_query_patch']:.4f}")
    print(f"  CW gap                 = {cs['mean_cw_gap']:+.4f}")
    if bm25_out:
        print(f"  BM25 ratio (D/P)       = {bm25_out['bm25_ratio_disc_over_patch']:.3f}x")


if __name__ == "__main__":
    main()
