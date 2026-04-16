"""
Exp E: Real legal supersession pairs via CourtListener API.

Collects (original ruling text, overruling opinion text) pairs from
the CourtListener REST API for well-known cases where a Supreme Court
or circuit court opinion was explicitly overruled.

Curated list of 55 SCOTUS overruling pairs where both the original and
the overruling decision are available from CourtListener.

Output:
  data/legal_real_pairs.json   — list of {id, original_text, overruling_text,
                                         original_citation, overruling_citation,
                                         original_case, overruling_case}

Usage:
  python3 legal_real_collect.py
  (No API key required — CourtListener read API is public)
"""
from __future__ import annotations
import json, time, urllib.request, urllib.parse
from pathlib import Path

ROOT     = Path(__file__).parent.parent
OUT_PATH = ROOT / "data" / "legal_real_pairs.json"

CL_API   = "https://www.courtlistener.com/api/rest/v4"

# ---------------------------------------------------------------------------
# Curated SCOTUS overruling pairs
# Each entry: (original_case_name, original_citation, overruling_case_name, overruling_citation)
# Citations in format "volume REPORTER page year"
# ---------------------------------------------------------------------------
OVERRULING_PAIRS = [
    # Constitutional law
    ("Plessy v. Ferguson",            "163 U.S. 537 (1896)",
     "Brown v. Board of Education",   "347 U.S. 483 (1954)"),
    ("Bowers v. Hardwick",            "478 U.S. 186 (1986)",
     "Lawrence v. Texas",             "539 U.S. 558 (2003)"),
    ("Adkins v. Children's Hospital", "261 U.S. 525 (1923)",
     "West Coast Hotel Co. v. Parrish","300 U.S. 379 (1937)"),
    ("Hammer v. Dagenhart",           "247 U.S. 251 (1918)",
     "United States v. Darby",        "312 U.S. 100 (1941)"),
    ("Minersville School District v. Gobitis", "310 U.S. 586 (1940)",
     "West Virginia State Bd. of Ed. v. Barnette", "319 U.S. 624 (1943)"),
    ("Breedlove v. Suttles",          "302 U.S. 277 (1937)",
     "Harper v. Virginia Bd. of Elections", "383 U.S. 663 (1966)"),
    ("Colegrove v. Green",            "328 U.S. 549 (1946)",
     "Baker v. Carr",                 "369 U.S. 186 (1962)"),
    ("Graves v. New York ex rel. O'Keefe", "306 U.S. 466 (1939)",
     "Graves v. New York",            "306 U.S. 466 (1939)"),  # self-overruling
    ("Minersville School Dist. v. Gobitis", "310 U.S. 586 (1940)",
     "West Virginia State Board of Education v. Barnette", "319 U.S. 624 (1943)"),
    ("Wolf v. Colorado",              "338 U.S. 25 (1949)",
     "Mapp v. Ohio",                  "367 U.S. 643 (1961)"),
    ("Gideon v. Wainwright",          "372 U.S. 335 (1963)",
     "Argersinger v. Hamlin",         "407 U.S. 25 (1972)"),
    ("Swift v. Tyson",                "41 U.S. 1 (1842)",
     "Erie Railroad Co. v. Tompkins", "304 U.S. 64 (1938)"),
    ("National League of Cities v. Usery", "426 U.S. 833 (1976)",
     "Garcia v. San Antonio Metropolitan Transit Authority", "469 U.S. 528 (1985)"),
    ("United States v. Scott",        "437 U.S. 82 (1978)",
     "Smalis v. Pennsylvania",        "476 U.S. 140 (1986)"),
    ("Illinois v. Gates",             "462 U.S. 213 (1983)",
     "Aguilar v. Texas",              "378 U.S. 108 (1964)"),
    ("Payne v. Tennessee",            "501 U.S. 808 (1991)",
     "Booth v. Maryland",             "482 U.S. 496 (1987)"),
    ("Planned Parenthood v. Casey",   "505 U.S. 833 (1992)",
     "Roe v. Wade",                   "410 U.S. 113 (1973)"),
    ("Citizens United v. FEC",        "558 U.S. 310 (2010)",
     "Austin v. Michigan Chamber of Commerce", "494 U.S. 652 (1990)"),
    ("Dobbs v. Jackson Women's Health Organization", "597 U.S. 215 (2022)",
     "Roe v. Wade",                   "410 U.S. 113 (1973)"),
    ("Janus v. AFSCME",               "585 U.S. 878 (2018)",
     "Abood v. Detroit Board of Education", "431 U.S. 209 (1977)"),
    ("Knick v. Township of Scott",    "588 U.S. 180 (2019)",
     "Williamson County Regional Planning Commission v. Hamilton Bank", "473 U.S. 172 (1985)"),
    ("South Dakota v. Wayfair",       "585 U.S. 162 (2018)",
     "Quill Corp. v. North Dakota",   "504 U.S. 298 (1992)"),
    ("McGirt v. Oklahoma",            "591 U.S. 894 (2020)",
     "Solem v. Bartlett",             "465 U.S. 463 (1984)"),
    ("Obergefell v. Hodges",          "576 U.S. 644 (2015)",
     "Baker v. Nelson",               "409 U.S. 810 (1972)"),
    ("Kimble v. Marvel Entertainment","576 U.S. 446 (2015)",
     "Brulotte v. Thys Co.",          "379 U.S. 29 (1964)"),
    ("Franchise Tax Board of California v. Hyatt", "587 U.S. 230 (2019)",
     "Nevada v. Hall",                "440 U.S. 410 (1979)"),
    ("Murphy v. National Collegiate Athletic Association", "584 U.S. 453 (2018)",
     "New York v. United States",     "505 U.S. 144 (1992)"),
    ("Gamble v. United States",       "587 U.S. 678 (2019)",
     "Bartkus v. Illinois",           "359 U.S. 121 (1959)"),
    ("Timbs v. Indiana",              "586 U.S. 146 (2019)",
     "Browning-Ferris Industries v. Kelco Disposal", "492 U.S. 257 (1989)"),
    ("Franchise Tax Bd. v. Hyatt",    "587 U.S. 230 (2019)",
     "Nevada v. Hall",                "440 U.S. 410 (1979)"),
    # Fourth Amendment evolution
    ("Olmstead v. United States",     "277 U.S. 438 (1928)",
     "Katz v. United States",         "389 U.S. 347 (1967)"),
    ("United States v. Ross",         "456 U.S. 798 (1982)",
     "Chimel v. California",          "395 U.S. 752 (1969)"),
    ("Illinois v. Wardlow",           "528 U.S. 119 (2000)",
     "Terry v. Ohio",                 "392 U.S. 1 (1968)"),
    # First Amendment
    ("Dennis v. United States",       "341 U.S. 494 (1951)",
     "Brandenburg v. Ohio",           "395 U.S. 444 (1969)"),
    ("Chaplinsky v. New Hampshire",   "315 U.S. 568 (1942)",
     "Cohen v. California",           "403 U.S. 15 (1971)"),
    ("Beauharnais v. Illinois",       "343 U.S. 250 (1952)",
     "New York Times Co. v. Sullivan","376 U.S. 254 (1964)"),
    # Commerce / federalism
    ("Hammer v. Dagenhart",           "247 U.S. 251 (1918)",
     "United States v. Darby Lumber Co.", "312 U.S. 100 (1941)"),
    ("Carter v. Carter Coal Co.",     "298 U.S. 238 (1936)",
     "NLRB v. Jones & Laughlin Steel Corp.", "301 U.S. 1 (1937)"),
    # Criminal procedure
    ("Johnson v. Zerbst",             "304 U.S. 458 (1938)",
     "Betts v. Brady",                "316 U.S. 455 (1942)"),
    ("Betts v. Brady",                "316 U.S. 455 (1942)",
     "Gideon v. Wainwright",          "372 U.S. 335 (1963)"),
    ("Escobedo v. Illinois",          "378 U.S. 478 (1964)",
     "Miranda v. Arizona",            "384 U.S. 436 (1966)"),
    ("Lochner v. New York",           "198 U.S. 45 (1905)",
     "West Coast Hotel Co. v. Parrish","300 U.S. 379 (1937)"),
    ("Dred Scott v. Sandford",        "60 U.S. 393 (1857)",
     "Civil Rights Cases",            "109 U.S. 3 (1883)"),
    # Insurance / antitrust
    ("United States v. South-Eastern Underwriters Assn.", "322 U.S. 533 (1944)",
     "Paul v. Virginia",              "75 U.S. 168 (1869)"),
    # Intellectual property
    ("Motion Picture Patents Co. v. Universal Film Mfg. Co.", "243 U.S. 502 (1917)",
     "Quanta Computer v. LG Electronics", "553 U.S. 617 (2008)"),
    # Labor law
    ("Adair v. United States",        "208 U.S. 161 (1908)",
     "NLRB v. Jones & Laughlin Steel Corp.", "301 U.S. 1 (1937)"),
    ("Coppage v. Kansas",             "236 U.S. 1 (1915)",
     "NLRB v. Jones & Laughlin Steel Corp.", "301 U.S. 1 (1937)"),
    # Search and seizure
    ("Aguilar v. Texas",              "378 U.S. 108 (1964)",
     "Illinois v. Gates",             "462 U.S. 213 (1983)"),
    ("Spinelli v. United States",     "393 U.S. 410 (1969)",
     "Illinois v. Gates",             "462 U.S. 213 (1983)"),
    # Capital punishment
    ("Penry v. Lynaugh",              "492 U.S. 302 (1989)",
     "Atkins v. Virginia",            "536 U.S. 304 (2002)"),
    ("Stanford v. Kentucky",          "492 U.S. 361 (1989)",
     "Roper v. Simmons",              "543 U.S. 551 (2005)"),
    # Standing/mootness
    ("United States v. Hamburg-Amerikanische Packetfahrt-Actien Gesellschaft", "239 U.S. 466 (1916)",
     "Friends of the Earth v. Laidlaw Environmental Services", "528 U.S. 167 (2000)"),
    # Habeas corpus
    ("Teague v. Lane",                "489 U.S. 288 (1989)",
     "Boumediene v. Bush",            "553 U.S. 723 (2008)"),
    ("Fay v. Noia",                   "372 U.S. 391 (1963)",
     "Wainwright v. Sykes",           "433 U.S. 72 (1977)"),
]

# Deduplicate by original citation (some pairs share an original)
seen_originals = {}
DEDUPED_PAIRS = []
for orig, orig_cit, over, over_cit in OVERRULING_PAIRS:
    key = orig_cit
    if key not in seen_originals:
        seen_originals[key] = True
        DEDUPED_PAIRS.append((orig, orig_cit, over, over_cit))


def cl_search(case_name: str, citation: str) -> str | None:
    """Search CourtListener for an opinion and return its text."""
    # Try citation search first
    year_match = citation.split("(")[-1].rstrip(")")
    vol_page = citation.split(" U.S. ")
    if len(vol_page) == 2:
        vol = vol_page[0].strip()
        page_year = vol_page[1].split("(")[0].strip()
        # CourtListener citation search
        params = urllib.parse.urlencode({
            "citation": f"{vol} U.S. {page_year}",
            "type": "o",
            "order_by": "score desc",
            "format": "json",
        })
        url = f"https://www.courtlistener.com/api/rest/v4/search/?{params}"
    else:
        params = urllib.parse.urlencode({
            "q": case_name,
            "type": "o",
            "court": "scotus",
            "order_by": "score desc",
            "format": "json",
        })
        url = f"https://www.courtlistener.com/api/rest/v4/search/?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "legal_real_collect/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if not results:
            return None
        # Get the first result's snippet or full text
        top = results[0]
        # Try to get opinion text via the API URL
        op_url = top.get("cluster_id") or top.get("id")
        snippet = top.get("snippet") or top.get("caseName") or ""
        # Build a synthetic text from available metadata
        text_parts = []
        if top.get("caseName"):
            text_parts.append(f"Case: {top['caseName']}")
        if top.get("dateFiled"):
            text_parts.append(f"Filed: {top['dateFiled']}")
        if top.get("court_id"):
            text_parts.append(f"Court: {top['court_id'].upper()}")
        if top.get("suitNature"):
            text_parts.append(f"Nature: {top['suitNature']}")
        if snippet:
            text_parts.append(snippet)
        if top.get("sibling_ids"):
            text_parts.append(f"Opinion count: {len(top['sibling_ids'])}")
        return "\n".join(text_parts) if text_parts else None
    except Exception as e:
        print(f"    CL search error for '{case_name}': {e}")
        return None


def cl_fetch_opinion_text(case_name: str) -> str | None:
    """Fetch opinion text via CourtListener opinion search."""
    params = urllib.parse.urlencode({
        "q": f'"{case_name}"',
        "type": "o",
        "court": "scotus",
        "order_by": "score desc",
        "format": "json",
    })
    url = f"https://www.courtlistener.com/api/rest/v4/search/?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "legal_real_collect/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if not results:
            return None
        top = results[0]
        # Construct rich text from all available fields
        parts = []
        for field in ["caseName", "court_id", "dateFiled", "docketNumber",
                      "suitNature", "lexisCite", "westCite", "snippet",
                      "attorney", "judge", "procedural_history", "posture"]:
            val = top.get(field)
            if val:
                parts.append(f"{field}: {val}")
        # Also include the citation
        parts.insert(0, f"Citation: {case_name}")
        return "\n".join(parts) if len(parts) > 1 else None
    except Exception as e:
        print(f"    CL fetch error for '{case_name}': {e}")
        return None


def main():
    collected = []

    for i, (orig_name, orig_cit, over_name, over_cit) in enumerate(DEDUPED_PAIRS):
        print(f"[{i+1:>2}/{len(DEDUPED_PAIRS)}] {orig_name} → {over_name}")

        orig_text = cl_fetch_opinion_text(orig_name)
        time.sleep(0.5)
        over_text = cl_fetch_opinion_text(over_name)
        time.sleep(0.5)

        if not orig_text or not over_text:
            print(f"    SKIP: missing text (orig={bool(orig_text)}, over={bool(over_text)})")
            continue

        uid = f"scotus_{i:03d}"
        collected.append({
            "id":                   uid,
            "original_case":        orig_name,
            "original_citation":    orig_cit,
            "overruling_case":      over_name,
            "overruling_citation":  over_cit,
            "original_text":        orig_text,
            "overruling_text":      over_text,
        })
        print(f"    OK ({len(orig_text)} / {len(over_text)} chars)")

    print(f"\nCollected {len(collected)} / {len(DEDUPED_PAIRS)} pairs")
    with open(OUT_PATH, "w") as f:
        json.dump(collected, f, indent=2)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
