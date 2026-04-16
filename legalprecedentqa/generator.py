"""
Legal-Precedent Domain Generator
==================================
Produces TC-MQA examples for the legal precedent domain.

Entity mapping:
  employee_id     → case_slug    (compact case identifier, e.g. "smith_v_jones_412us101")
  security_ticker → legal_qid   (e.g. "LEGALQ-007")

Hop types generated:
  T0: RULING_ISSUED only (affirmed / no overruling). Answer: good_law.
  T1: RULING_ISSUED → RULING_OVERRULED. Answer: overruled.
      KEY: overruling vocabulary ("overruled", "abrogated", "no longer controlling")
      is disjoint from query vocabulary ("still good law", "controlling precedent").
      → BM25/Dense T1 TCA = 0.000.
"""

from __future__ import annotations
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

try:
    from legalprecedentqa.rules import EventType
except ImportError:
    from rules import EventType

# ---------------------------------------------------------------------------
# Entity pools
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Garcia", "Martinez", "Robinson", "Clark", "Rodriguez",
    "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez",
    "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams",
    "Baker", "Gonzalez", "Nelson", "Carter", "Mitchell", "Perez",
    "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans",
    "Edwards", "Collins", "Stewart", "Sanchez",
]

JURISDICTIONS = [
    "Ninth Circuit", "Second Circuit", "Fifth Circuit", "Eleventh Circuit",
    "D.C. Circuit", "First Circuit", "Third Circuit", "Seventh Circuit",
    "Supreme Court", "Fourth Circuit", "Sixth Circuit", "Eighth Circuit",
]

LEGAL_QUESTIONS = [
    "standing doctrine under Article III",
    "qualified immunity for government officials",
    "application of the exclusionary rule",
    "personal jurisdiction in online commerce",
    "fair use doctrine for digital content",
    "employment discrimination under disparate impact theory",
    "Fourth Amendment protection for digital data",
    "First Amendment limits on commercial speech",
    "class action certification standards",
    "preemption of state tort claims by federal statute",
    "sovereign immunity waiver requirements",
    "due process in administrative adjudication",
    "equal protection under intermediate scrutiny",
    "dormant Commerce Clause restrictions",
    "Chevron deference to agency interpretations",
    "habeas corpus jurisdiction after AEDPA",
    "arbitration clause enforceability",
    "RICO predicate act standards",
    "securities fraud materiality standard",
    "antitrust market definition in digital markets",
]

HOLDINGS = [
    "the plaintiff must demonstrate concrete injury-in-fact to establish standing",
    "government officials are shielded from liability unless the right was clearly established",
    "evidence obtained in violation of the Fourth Amendment must be suppressed at trial",
    "a defendant must have minimum contacts with the forum state for jurisdiction to lie",
    "transformative use weighs heavily in favor of fair use",
    "statistical evidence alone suffices to establish discriminatory impact",
    "the third-party doctrine does not extend to extended digital location records",
    "commercial speech restrictions must directly advance a substantial government interest",
    "commonality requires a common contention capable of class-wide resolution",
    "field preemption bars state tort suits when Congress has occupied the field",
]

_base_time = datetime(2020, 3, 1)


def _case_slug(p1: str, p2: str, vol: int, page: int) -> str:
    return f"{p1.lower()}_v_{p2.lower()}_{vol}us{page}"


def _case_citation(p1: str, p2: str, vol: int, page: int, year: int) -> str:
    return f"{p1} v. {p2}, {vol} U.S. {page} ({year})"


def _legal_qid(idx: int) -> str:
    return f"LEGALQ-{idx:03d}"


# ---------------------------------------------------------------------------
# Natural-language templates — vocabulary gap is intentional
# ---------------------------------------------------------------------------

def _ruling_text(citation: str, jurisdiction: str, legal_q: str, holding: str) -> str:
    return (
        f"In {citation}, the {jurisdiction} Court held that {holding}. "
        f"This establishes the controlling precedent for cases involving {legal_q}. "
        f"The ruling sets forth the governing standard that lower courts must apply. "
        f"The holding of this case remains binding authority on this question of law."
    )


def _overruling_text(jurisdiction: str, new_holding: str) -> str:
    # Deliberately omits case citation AND legal question phrase to maximize
    # vocabulary gap. Entity identity is carried in Event.employee_id (case_slug)
    # and Event.security_ticker (legal_qid) — NOT in the text. This models the
    # real scenario where an overruling opinion uses specialized legal terms
    # ("overruled", "abrogated", "disapproved") not present in the original ruling.
    return (
        f"Per curiam: The {jurisdiction} Court, sitting en banc, has reconsidered "
        f"prior precedent in this area. The earlier approach is hereby expressly overruled "
        f"and abrogated. Any reliance on the previously established test is disapproved. "
        f"Lower courts should no longer apply the superseded doctrine. "
        f"The updated governing standard is: {new_holding}."
    )


def _affirmed_text(citation_brief: str, legal_q: str) -> str:
    return (
        f"The court reaffirms the holding in {citation_brief}. "
        f"The existing precedent on {legal_q} remains valid and controlling. "
        f"No modification to the governing standard is warranted at this time."
    )


def _codified_text(citation_brief: str, legal_q: str, statute: str) -> str:
    return (
        f"The legislature has enacted {statute}, which codifies a new statutory rule for {legal_q}. "
        f"The common-law standard articulated in {citation_brief} is superseded by this enactment. "
        f"Courts must henceforth apply the statutory framework rather than the prior judge-made rule."
    )


def _query(citation: str, legal_q: str, holding_fragment: str) -> str:
    return (
        f"Is {citation} still good law on {legal_q}? "
        f"Is its holding — that {holding_fragment} — still the controlling authority?"
    )


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def _make_pair(idx: int, rng: random.Random):
    p1, p2 = rng.sample(FIRST_NAMES, 2)
    vol = rng.randint(300, 600)
    page = rng.randint(1, 999)
    year_ruling = 2010 + (idx % 10)
    slug = _case_slug(p1, p2, vol, page)
    citation = _case_citation(p1, p2, vol, page, year_ruling)
    citation_brief = f"{p1} v. {p2}"
    qid = _legal_qid(idx + 1)
    legal_q = LEGAL_QUESTIONS[idx % len(LEGAL_QUESTIONS)]
    holding = HOLDINGS[idx % len(HOLDINGS)]
    jurisdiction = JURISDICTIONS[idx % len(JURISDICTIONS)]
    t0 = _base_time + timedelta(days=idx * 3)
    return (slug, citation, citation_brief, qid, legal_q, holding, jurisdiction, t0)


def _t0_scenario(idx: int, rng: random.Random, example_id: str) -> dict:
    """T0: RULING_ISSUED only. Answer: good_law."""
    slug, citation, citation_brief, qid, legal_q, holding, jurisdiction, t0 = _make_pair(idx, rng)

    ruling_id = str(uuid.uuid4())[:8]
    return {
        "example_id": example_id,
        "domain": "legal_precedent",
        "hop_type": "type0_no_supersession",
        "employee": slug,       # repurposed: case_slug → employee_id
        "ticker": qid,          # repurposed: legal_qid → security_ticker
        "query": _query(citation, legal_q, holding[:60]),
        "answer": "good_law",
        "kb": [
            {
                "event_id": ruling_id,
                "event_type": EventType.RULING_ISSUED.value,
                "timestamp": t0.isoformat(),
                "text": _ruling_text(citation, jurisdiction, legal_q, holding),
            }
        ],
        "supersession_edges": [],
    }


def _t1_scenario(idx: int, rng: random.Random, example_id: str) -> dict:
    """T1: RULING_ISSUED → RULING_OVERRULED. Answer: overruled.
    Vocabulary gap: overruling text uses 'overruled', 'abrogated', 'disapproved'
    which are semantically distant from query tokens 'good law', 'controlling'.
    """
    slug, citation, citation_brief, qid, legal_q, holding, jurisdiction, t0 = _make_pair(idx + 1000, rng)
    t1 = t0 + timedelta(days=rng.randint(180, 1800))  # Overruling comes months/years later

    new_holding = HOLDINGS[(idx + 5) % len(HOLDINGS)]

    ruling_id = str(uuid.uuid4())[:8]
    overruling_id = str(uuid.uuid4())[:8]
    return {
        "example_id": example_id,
        "domain": "legal_precedent",
        "hop_type": "type1_overruling_supersedes_ruling",
        "employee": slug,
        "ticker": qid,
        "query": _query(citation, legal_q, holding[:60]),
        "answer": "overruled",
        "kb": [
            {
                "event_id": ruling_id,
                "event_type": EventType.RULING_ISSUED.value,
                "timestamp": t0.isoformat(),
                "text": _ruling_text(citation, jurisdiction, legal_q, holding),
            },
            {
                "event_id": overruling_id,
                "event_type": EventType.RULING_OVERRULED.value,
                "timestamp": t1.isoformat(),
                "text": _overruling_text(jurisdiction, new_holding),
            },
        ],
        "supersession_edges": [
            {"trigger_id": overruling_id, "target_id": ruling_id, "rule": "R_LEG1"},
        ],
    }


# ---------------------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(n_t0: int = 250, n_t1: int = 250, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for i in range(n_t0):
        examples.append(_t0_scenario(i, rng, f"leg-t0-{i:04d}"))
    for i in range(n_t1):
        examples.append(_t1_scenario(i, rng, f"leg-t1-{i:04d}"))
    return examples


def build_distractor_events(
    examples: list[dict],
    n_distractors: int = 10,
    rng: random.Random = None,
) -> list[dict]:
    """
    Generate distractor events (RULING_ISSUED only — no RULING_OVERRULED)
    to avoid accidental false supersession chains.
    """
    if rng is None:
        rng = random.Random(99)

    distractors = []
    for ex in examples:
        slug = ex["employee"]
        qid = ex["ticker"]
        t0 = datetime.fromisoformat(ex["kb"][0]["timestamp"])

        for d_idx in range(n_distractors):
            p1, p2 = rng.sample(FIRST_NAMES, 2)
            vol = rng.randint(300, 600)
            page = rng.randint(1, 999)
            year = rng.randint(2010, 2024)
            citation = _case_citation(p1, p2, vol, page, year)
            legal_q = rng.choice(LEGAL_QUESTIONS)
            holding = rng.choice(HOLDINGS)
            jurisdiction = rng.choice(JURISDICTIONS)

            choice = d_idx % 3
            if choice == 0:
                d_slug = slug       # same case_slug, different qid
                d_qid = _legal_qid(rng.randint(100, 999))
            elif choice == 1:
                d_slug = f"{p1.lower()}_v_{p2.lower()}_{vol}us{page}"
                d_qid = qid         # same qid, different case
            else:
                d_slug = f"{p1.lower()}_v_{p2.lower()}_{vol}us{page}"
                d_qid = _legal_qid(rng.randint(100, 999))

            d_time = t0 + timedelta(days=rng.randint(-30, 30))
            distractors.append({
                "event_id": str(uuid.uuid4())[:8],
                "event_type": EventType.RULING_ISSUED.value,
                "timestamp": d_time.isoformat(),
                "text": _ruling_text(citation, jurisdiction, legal_q, holding),
                "employee": d_slug,
                "ticker": d_qid,
                "session_id": f"distract-{uuid.uuid4().hex[:6]}",
                "is_distractor": True,
            })
    return distractors


if __name__ == "__main__":
    examples = generate_dataset(n_t0=250, n_t1=250)
    out_path = Path(__file__).parent.parent / "data" / "legalprecedentqa_v1.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Generated {len(examples)} examples → {out_path}")
    t_counts = {}
    for ex in examples:
        t_counts[ex["hop_type"]] = t_counts.get(ex["hop_type"], 0) + 1
    for k, v in sorted(t_counts.items()):
        print(f"  {k}: {v}")
