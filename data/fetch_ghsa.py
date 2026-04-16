#!/usr/bin/env python3
"""
Fetch real GHSA advisories and their corresponding GitHub release notes.
Saves tuples of (disclosure_text, patch_text, product, cve_id) for retrieval benchmark.

References field is a list of plain strings (not dicts).
cve_id is a top-level field in the advisory JSON.
"""

import json
import time
import re
import urllib.request
import urllib.error
import urllib.parse
import os
import sys

OUTPUT_PATH = "./data/ghsa_real_pairs.json"
CHECKPOINT_PATH = "./data/ghsa_checkpoint.json"
SLEEP_BETWEEN = 2.0  # seconds between API calls

def make_request(url):
    """Make a GET request and return parsed JSON, or None on error."""
    req_headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "academic-research-ghsa-fetch/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
            else:
                print(f"  HTTP {resp.status} for {url}", file=sys.stderr)
                return None
    except urllib.error.HTTPError as e:
        print(f"  HTTPError {e.code} for {url}: {e.reason}", file=sys.stderr)
        if e.code == 429:
            print("  RATE LIMIT HIT — stopping.", file=sys.stderr)
            return "RATE_LIMIT"
        return None
    except Exception as e:
        print(f"  Error fetching {url}: {e}", file=sys.stderr)
        return None


def extract_release_url(references):
    """
    Find a GitHub releases/tag URL in the references list.
    References is a list of plain URL strings.
    """
    for url in references:
        if isinstance(url, str) and re.search(
            r"github\.com/[^/]+/[^/]+/releases/tag/[^/\s]+", url
        ):
            return url
    return None


def parse_release_url(url):
    """Parse owner, repo, tag from a GitHub releases URL."""
    m = re.search(r"github\.com/([^/]+)/([^/]+)/releases/tag/(.+)$", url)
    if m:
        owner = m.group(1)
        repo = m.group(2)
        tag = m.group(3).split("?")[0].split("#")[0]
        return owner, repo, tag
    return None, None, None


def fetch_release_body(owner, repo, tag):
    """Fetch the release note body from GitHub API."""
    tag_encoded = urllib.parse.quote(tag, safe="")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag_encoded}"
    data = make_request(api_url)
    if data == "RATE_LIMIT":
        return "RATE_LIMIT"
    if data and isinstance(data, dict):
        return data.get("body", "")
    return None


def get_product_name(advisory):
    """Extract product/package name from advisory vulnerabilities."""
    vulns = advisory.get("vulnerabilities", [])
    if vulns:
        pkg = vulns[0].get("package", {})
        name = pkg.get("name", "")
        if name:
            return name
    # Fall back to repo name from source_code_location
    src = advisory.get("source_code_location", "")
    if src:
        m = re.search(r"github\.com/[^/]+/([^/\s]+)", src)
        if m:
            return m.group(1)
    # Fall back to repo name from references
    refs = advisory.get("references", [])
    rel_url = extract_release_url(refs)
    if rel_url:
        m = re.search(r"github\.com/[^/]+/([^/]+)/releases", rel_url)
        if m:
            return m.group(1)
    return advisory.get("ghsa_id", "unknown")


def is_english(text):
    """Rough heuristic: check ASCII ratio."""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > 0.80


def load_checkpoint():
    """Load any existing checkpoint data."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"pairs": [], "processed_ghsa_ids": []}


def save_checkpoint(data):
    """Save checkpoint to disk."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main():
    checkpoint = load_checkpoint()
    pairs = checkpoint["pairs"]
    processed_ids = set(checkpoint["processed_ghsa_ids"])
    print(f"Resuming from checkpoint: {len(pairs)} pairs, {len(processed_ids)} processed advisories")

    stats = {
        "advisories_fetched": 0,
        "had_release_url": 0,
        "release_fetch_attempted": 0,
        "release_body_found": 0,
        "passed_length_filter": 0,
        "passed_no_cve_filter": 0,
        "passed_english_filter": 0,
        "final_kept": len(pairs),
    }

    # Severities to sweep
    severities = ["critical", "high", "medium"]
    pages_per_severity = 4  # up to 400 advisories per severity

    rate_limited = False

    for severity in severities:
        if rate_limited:
            break
        for page in range(1, pages_per_severity + 1):
            if rate_limited:
                break
            if stats["final_kept"] >= 120:
                print(f"Reached 120 pairs, stopping.")
                break

            url = (
                f"https://api.github.com/advisories"
                f"?type=reviewed&severity={severity}&per_page=100&page={page}"
            )
            print(f"\nFetching {severity} page {page}: {url}")
            advisories = make_request(url)

            if advisories == "RATE_LIMIT":
                rate_limited = True
                break
            if not advisories:
                print(f"  No data returned, skipping.")
                break
            if not isinstance(advisories, list) or len(advisories) == 0:
                print(f"  Empty list, done with {severity}.")
                break

            stats["advisories_fetched"] += len(advisories)
            print(f"  Got {len(advisories)} advisories")
            time.sleep(SLEEP_BETWEEN)

            for adv in advisories:
                if rate_limited:
                    break
                if stats["final_kept"] >= 120:
                    break

                ghsa_id = adv.get("ghsa_id", "")
                if ghsa_id in processed_ids:
                    continue
                processed_ids.add(ghsa_id)

                # Get CVE ID — top-level field or from identifiers
                cve_id = adv.get("cve_id", "")
                if not cve_id:
                    for ident in adv.get("identifiers", []):
                        if ident.get("type") == "CVE":
                            cve_id = ident.get("value", "")
                            break
                if not cve_id:
                    continue

                # Get disclosure text
                disclosure_text = adv.get("description", "") or adv.get("summary", "")
                if not disclosure_text or len(disclosure_text) < 50:
                    continue

                # Find release URL in references (list of plain strings)
                refs = adv.get("references", [])
                release_url = extract_release_url(refs)
                if not release_url:
                    continue

                stats["had_release_url"] += 1
                owner, repo, tag = parse_release_url(release_url)
                if not owner:
                    continue

                print(f"  [{ghsa_id}] {cve_id} — {owner}/{repo} tag={tag}")

                # Fetch release note
                stats["release_fetch_attempted"] += 1
                time.sleep(SLEEP_BETWEEN)
                body = fetch_release_body(owner, repo, tag)

                if body == "RATE_LIMIT":
                    rate_limited = True
                    break
                if body is None:
                    print(f"    No release found at API")
                    continue
                if not body:
                    print(f"    Empty release body")
                    continue

                stats["release_body_found"] += 1

                # Filter: length > 100
                if len(body.strip()) <= 100:
                    print(f"    Too short ({len(body)} chars)")
                    continue
                stats["passed_length_filter"] += 1

                # Filter: body must NOT contain the CVE ID
                if cve_id in body:
                    print(f"    Patch text contains CVE ID — skipped (vocabulary gap test failed)")
                    continue
                stats["passed_no_cve_filter"] += 1

                # Filter: English
                if not is_english(body):
                    print(f"    Non-English body — skipped")
                    continue
                stats["passed_english_filter"] += 1

                # Extract product name
                product = get_product_name(adv)

                pair = {
                    "id": ghsa_id,
                    "cve_id": cve_id,
                    "product": product,
                    "disclosure_text": disclosure_text,
                    "patch_text": body,
                    "release_url": release_url,
                    "patch_contains_cve_id": False,
                    "severity": severity,
                    "owner": owner,
                    "repo": repo,
                    "tag": tag,
                }
                pairs.append(pair)
                stats["final_kept"] += 1
                print(
                    f"    KEPT ({stats['final_kept']} total). "
                    f"Disclosure: {len(disclosure_text)} chars, Patch: {len(body)} chars"
                )

                # Save checkpoint after each kept pair
                save_checkpoint({"pairs": pairs, "processed_ghsa_ids": list(processed_ids)})

            # Save after each page
            save_checkpoint({"pairs": pairs, "processed_ghsa_ids": list(processed_ids)})

    # Final output save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    print("\n" + "="*60)
    print("DONE")
    print(f"  Advisories fetched:          {stats['advisories_fetched']}")
    print(f"  Had GitHub release URL:       {stats['had_release_url']}")
    print(f"  Release fetch attempted:      {stats['release_fetch_attempted']}")
    print(f"  Release body found:           {stats['release_body_found']}")
    print(f"  Passed length filter:         {stats['passed_length_filter']}")
    print(f"  Passed no-CVE-ID filter:      {stats['passed_no_cve_filter']}")
    print(f"  Passed English filter:        {stats['passed_english_filter']}")
    print(f"  Final pairs saved:            {stats['final_kept']}")
    print(f"  Output: {OUTPUT_PATH}")
    print("="*60)

    if rate_limited:
        print("\nWARNING: Rate limit hit. Partial data saved. Re-run to continue from checkpoint.")

    return stats


if __name__ == "__main__":
    main()
