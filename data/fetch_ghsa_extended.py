#!/usr/bin/env python3
"""
Extended GHSA fetch — continues from checkpoint, sweeps more pages and severities.
Adds deduplication on release_url to reduce same-release-note concentration.
Also tries constructing release API URLs from source_code_location + first_patched_version.
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
SLEEP_BETWEEN = 2.0

def make_request(url):
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
                return None
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"  RATE LIMIT HIT — stopping.", file=sys.stderr)
            return "RATE_LIMIT"
        if e.code == 404:
            return None
        print(f"  HTTPError {e.code} for {url}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error fetching {url}: {e}", file=sys.stderr)
        return None


def extract_release_url(references):
    """References is a list of plain URL strings."""
    for url in references:
        if isinstance(url, str) and re.search(
            r"github\.com/[^/]+/[^/]+/releases/tag/[^/\s]+", url
        ):
            return url
    return None


def parse_release_url(url):
    m = re.search(r"github\.com/([^/]+)/([^/]+)/releases/tag/(.+)$", url)
    if m:
        owner = m.group(1)
        repo = m.group(2)
        tag = m.group(3).split("?")[0].split("#")[0]
        return owner, repo, tag
    return None, None, None


def fetch_release_by_tag(owner, repo, tag):
    """Fetch release note body by tag."""
    tag_encoded = urllib.parse.quote(tag, safe="")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag_encoded}"
    data = make_request(api_url)
    if data == "RATE_LIMIT":
        return "RATE_LIMIT"
    if data and isinstance(data, dict):
        return data.get("body", "")
    return None


def try_version_as_tag(owner, repo, version):
    """
    Try various tag formats from a version string.
    e.g. "1.2.3" -> try "v1.2.3", "1.2.3", "release-1.2.3"
    """
    candidates = [
        version,
        f"v{version}",
        f"release-{version}",
        f"release/{version}",
        version.replace(".", "-"),
    ]
    for tag in candidates:
        time.sleep(0.5)
        body = fetch_release_by_tag(owner, repo, tag)
        if body == "RATE_LIMIT":
            return "RATE_LIMIT", tag
        if body is not None and body != "":
            return body, tag
    return None, None


def get_product_name(advisory):
    vulns = advisory.get("vulnerabilities", [])
    if vulns:
        pkg = vulns[0].get("package", {})
        name = pkg.get("name", "")
        if name:
            return name
    src = advisory.get("source_code_location", "")
    if src:
        m = re.search(r"github\.com/[^/]+/([^/\s]+)", src)
        if m:
            return m.group(1)
    return advisory.get("ghsa_id", "unknown")


def get_cve_id(advisory):
    cve_id = advisory.get("cve_id", "")
    if cve_id:
        return cve_id
    for ident in advisory.get("identifiers", []):
        if ident.get("type") == "CVE":
            return ident.get("value", "")
    return None


def is_english(text):
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > 0.80


def get_source_code_owner_repo(advisory):
    src = advisory.get("source_code_location", "")
    if src:
        m = re.search(r"github\.com/([^/]+)/([^/\s]+?)(?:\.git)?$", src)
        if m:
            return m.group(1), m.group(2)
    return None, None


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"pairs": [], "processed_ghsa_ids": []}


def save_checkpoint(data):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main():
    checkpoint = load_checkpoint()
    pairs = checkpoint["pairs"]
    processed_ids = set(checkpoint["processed_ghsa_ids"])

    # Track seen release URLs for dedup
    seen_release_urls = set(p["release_url"] for p in pairs)
    # Track repo diversity: max pairs per repo
    from collections import Counter
    repo_counts = Counter(p["repo"] for p in pairs)
    MAX_PER_REPO = 3  # cap same-repo to 3 pairs for diversity

    print(f"Loaded {len(pairs)} pairs, {len(processed_ids)} processed IDs, {len(seen_release_urls)} release URLs")

    stats = {
        "advisories_fetched": 0,
        "had_release_url": 0,
        "tried_version_fallback": 0,
        "release_body_found": 0,
        "passed_length_filter": 0,
        "passed_no_cve_filter": 0,
        "passed_english_filter": 0,
        "dedup_skipped": 0,
        "repo_cap_skipped": 0,
        "final_kept": len(pairs),
    }

    severities = ["critical", "high", "medium", "low"]
    pages_per_severity = 6  # 600 per severity

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
            print(f"\n[{severity} p{page}] Fetching: {url}")
            advisories = make_request(url)

            if advisories == "RATE_LIMIT":
                rate_limited = True
                break
            if not advisories or not isinstance(advisories, list):
                print(f"  No data / end of results.")
                break
            if len(advisories) == 0:
                print(f"  Empty, done with {severity}.")
                break

            stats["advisories_fetched"] += len(advisories)
            print(f"  Got {len(advisories)} advisories")
            time.sleep(SLEEP_BETWEEN)

            for adv in advisories:
                if rate_limited or stats["final_kept"] >= 120:
                    break

                ghsa_id = adv.get("ghsa_id", "")
                if ghsa_id in processed_ids:
                    continue
                processed_ids.add(ghsa_id)

                cve_id = get_cve_id(adv)
                if not cve_id:
                    continue

                disclosure_text = adv.get("description", "") or adv.get("summary", "")
                if not disclosure_text or len(disclosure_text) < 50:
                    continue

                refs = adv.get("references", [])
                release_url = extract_release_url(refs)

                body = None
                actual_tag = None
                owner, repo = None, None

                if release_url:
                    stats["had_release_url"] += 1
                    owner, repo, tag = parse_release_url(release_url)
                    if owner:
                        time.sleep(SLEEP_BETWEEN)
                        body = fetch_release_by_tag(owner, repo, tag)
                        actual_tag = tag
                        if body == "RATE_LIMIT":
                            rate_limited = True
                            break
                else:
                    # Try to construct release URL from source_code_location + first_patched_version
                    vulns = adv.get("vulnerabilities", [])
                    if vulns:
                        version = vulns[0].get("first_patched_version", "")
                        if version:
                            src_owner, src_repo = get_source_code_owner_repo(adv)
                            if src_owner:
                                stats["tried_version_fallback"] += 1
                                owner, repo = src_owner, src_repo
                                time.sleep(SLEEP_BETWEEN)
                                body, actual_tag = try_version_as_tag(owner, repo, version)
                                if body == "RATE_LIMIT":
                                    rate_limited = True
                                    break
                                if body and actual_tag:
                                    release_url = f"https://github.com/{owner}/{repo}/releases/tag/{actual_tag}"

                if not body or body is None:
                    continue

                stats["release_body_found"] += 1

                # Dedup by release URL
                if release_url and release_url in seen_release_urls:
                    # Allow a few per release_url if different CVEs
                    existing_count = sum(1 for p in pairs if p["release_url"] == release_url)
                    if existing_count >= 3:
                        stats["dedup_skipped"] += 1
                        continue

                # Cap per repo
                if repo and repo_counts.get(repo, 0) >= MAX_PER_REPO:
                    stats["repo_cap_skipped"] += 1
                    continue

                # Length filter
                if len(body.strip()) <= 100:
                    continue
                stats["passed_length_filter"] += 1

                # No CVE ID filter
                if cve_id in body:
                    continue
                stats["passed_no_cve_filter"] += 1

                # English filter
                if not is_english(body):
                    continue
                stats["passed_english_filter"] += 1

                product = get_product_name(adv)

                pair = {
                    "id": ghsa_id,
                    "cve_id": cve_id,
                    "product": product,
                    "disclosure_text": disclosure_text,
                    "patch_text": body,
                    "release_url": release_url or f"https://github.com/{owner}/{repo}/releases/tag/{actual_tag}",
                    "patch_contains_cve_id": False,
                    "severity": severity,
                    "owner": owner,
                    "repo": repo,
                    "tag": actual_tag,
                }
                pairs.append(pair)
                seen_release_urls.add(release_url)
                repo_counts[repo] = repo_counts.get(repo, 0) + 1
                stats["final_kept"] += 1
                print(
                    f"  KEPT [{ghsa_id}] {cve_id} {product} "
                    f"({stats['final_kept']} total, patch={len(body)}c)"
                )

                save_checkpoint({"pairs": pairs, "processed_ghsa_ids": list(processed_ids)})

            save_checkpoint({"pairs": pairs, "processed_ghsa_ids": list(processed_ids)})

    with open(OUTPUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    print("\n" + "="*60)
    print("EXTENDED FETCH DONE")
    print(f"  Advisories fetched:          {stats['advisories_fetched']}")
    print(f"  Had GitHub release URL:       {stats['had_release_url']}")
    print(f"  Tried version fallback:       {stats['tried_version_fallback']}")
    print(f"  Release body found:           {stats['release_body_found']}")
    print(f"  Passed length filter:         {stats['passed_length_filter']}")
    print(f"  Passed no-CVE-ID filter:      {stats['passed_no_cve_filter']}")
    print(f"  Passed English filter:        {stats['passed_english_filter']}")
    print(f"  Dedup skipped:                {stats['dedup_skipped']}")
    print(f"  Repo cap skipped:             {stats['repo_cap_skipped']}")
    print(f"  Final pairs saved:            {stats['final_kept']}")
    print(f"  Output: {OUTPUT_PATH}")
    print("="*60)

    if rate_limited:
        print("\nWARNING: Rate limit hit. Partial data saved.")

    return stats


if __name__ == "__main__":
    main()
