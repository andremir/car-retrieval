"""
Expand GHSA real pairs using authenticated GitHub CLI token.
Uses source_code_location + first_patched_version to construct release URLs.
Targets 300+ pairs. Merges with existing ghsa_real_pairs.json.
"""
import json, re, time, subprocess, urllib.request, urllib.parse
from pathlib import Path

OUT = Path(__file__).parent.parent / "data" / "ghsa_real_pairs.json"
existing = json.load(open(OUT)) if OUT.exists() else []
seen_ids = {p["id"] for p in existing}
seen_cve = {p["cve_id"] for p in existing}
print(f"Starting with {len(existing)} existing pairs")

token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

def api_get(url):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception:
        return None

def try_release_note(owner: str, repo: str, version: str) -> tuple[str | None, str | None]:
    """Try v{version}, then {version}, then search releases."""
    for tag in [f"v{version}", version, f"release-{version}", f"{repo}-{version}"]:
        enc = urllib.parse.quote(tag, safe="")
        data = api_get(f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{enc}")
        if data and data.get("body") and len(data["body"]) > 80:
            return data["body"], f"https://github.com/{owner}/{repo}/releases/tag/{tag}"
        time.sleep(0.1)
    return None, None

new_pairs = []
TARGET = 350

for severity in ["critical", "high", "medium", "low"]:
    if len(existing) + len(new_pairs) >= TARGET:
        break
    for page in range(1, 11):
        if len(existing) + len(new_pairs) >= TARGET:
            break
        url = (f"https://api.github.com/advisories?"
               f"type=reviewed&per_page=100&severity={severity}&page={page}")
        data = api_get(url)
        if not data:
            break
        if not isinstance(data, list) or len(data) == 0:
            break

        added = 0
        for adv in data:
            ghsa_id = adv.get("ghsa_id", "")
            if ghsa_id in seen_ids:
                continue
            cve_id = adv.get("cve_id", "")
            if not cve_id or cve_id in seen_cve:
                continue

            # Get disclosure text
            disclosure = (adv.get("description") or adv.get("summary", "")).strip()
            if len(disclosure) < 80:
                continue

            # Get package name
            vulns = adv.get("vulnerabilities", [])
            if not vulns:
                continue
            pkg = vulns[0].get("package", {}).get("name", "")
            if not pkg:
                continue
            version = vulns[0].get("first_patched_version")
            if not version:
                continue

            # Get GitHub repo from source_code_location
            src = adv.get("source_code_location") or ""
            m_src = re.match(r"https://github\.com/([^/]+)/([^/]+)", src)

            # Also try references
            refs = adv.get("references", [])
            ref_release = next(
                (r for r in refs if isinstance(r, str) and
                 re.match(r"https://github\.com/[^/]+/[^/]+/releases/tag/", r)),
                None,
            )

            patch_text, release_url = None, None

            if ref_release:
                m_ref = re.match(r"https://github\.com/([^/]+)/([^/]+)/releases/tag/(.+)", ref_release)
                if m_ref:
                    data_r = api_get(
                        f"https://api.github.com/repos/{m_ref.group(1)}/{m_ref.group(2)}"
                        f"/releases/tags/{urllib.parse.quote(m_ref.group(3), safe='')}"
                    )
                    if data_r and data_r.get("body") and len(data_r["body"]) > 80:
                        patch_text = data_r["body"]
                        release_url = ref_release
                        time.sleep(0.15)

            if not patch_text and m_src:
                owner, repo = m_src.group(1), m_src.group(2)
                patch_text, release_url = try_release_note(owner, repo, version)

            if not patch_text:
                continue
            if cve_id.lower() in patch_text.lower():
                continue
            if len(patch_text.split()) < 20:
                continue

            m_url = re.match(r"https://github\.com/([^/]+)/([^/]+)/releases/tag/(.+)",
                             release_url) if release_url else None
            new_pairs.append({
                "id": ghsa_id,
                "cve_id": cve_id,
                "product": pkg,
                "disclosure_text": disclosure,
                "patch_text": patch_text,
                "release_url": release_url or "",
                "patch_contains_cve_id": False,
                "severity": severity,
                "owner": m_url.group(1) if m_url else "",
                "repo": m_url.group(2) if m_url else "",
                "tag": m_url.group(3) if m_url else "",
            })
            seen_ids.add(ghsa_id)
            seen_cve.add(cve_id)
            added += 1

        total = len(existing) + len(new_pairs)
        print(f"  [{severity} p{page}] checked {len(data)} advs, +{added} new, total={total}")
        time.sleep(0.5)

all_pairs = existing + new_pairs
print(f"\nFinal: {len(all_pairs)} pairs ({len(existing)} existing + {len(new_pairs)} new)")
with open(OUT, "w") as f:
    json.dump(all_pairs, f, indent=2)
print(f"Saved → {OUT}")
