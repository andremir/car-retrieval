"""
FinSuperQA Dataset Generator
==============================
Generates synthetic compliance conversation sessions with:
  1. Random but realistic event sequences (rule-governed)
  2. Rule-derived ground truth (no human annotation)
  3. Natural-language conversation turns expressing the events
  4. Multi-hop queries requiring temporal supersession reasoning

Output: JSONL where each line is one (KB_context, query, answer, provenance) example.

Multi-hop types generated:
  Type 1 — Simple supersession (2-hop): pre-clearance → blackout → NOT compliant
  Type 2 — Chain supersession (3-hop): pre-clearance → blackout → exception → compliant
  Type 3 — Cross-session supersession: events in 3 different sessions
"""

from __future__ import annotations
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rules import EventType, QueryType, ComplianceStatus, SUPERSESSION_RULES
from events import Event, KnowledgeBase
from supersession_graph import SupersessionGraph


# ---------------------------------------------------------------------------
# Entity pools (realistic but synthetic)
# Large enough that each example gets a UNIQUE (employee, ticker) pair.
# This is critical for eval correctness: entity+temporal probing of the full
# corpus must not accidentally retrieve events from a different example that
# happens to share the same (employee, ticker).
# Pool size: 100 × 60 = 6000 unique pairs >> 1000 examples.
# ---------------------------------------------------------------------------

_SURNAMES = [
    "Chen", "Patel", "Kim", "Singh", "Torres", "Novak", "Okonkwo", "Larsson",
    "Costa", "Zhao", "Müller", "Nakamura", "Ferreira", "Gupta", "Ahmed",
    "Johnson", "Williams", "Brown", "Jones", "Garcia", "Martinez", "Davis",
    "Miller", "Wilson", "Anderson", "Taylor", "Thomas", "Jackson", "White",
    "Harris", "Martin", "Thompson", "Robinson", "Clark", "Lewis", "Lee",
    "Walker", "Hall", "Allen", "Young", "Hernandez", "King", "Wright",
    "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez",
    "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips",
    "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez",
    "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy", "Bailey",
    "Rivera", "Cooper", "Richardson", "Cox", "Howard", "Ward", "Torres2",
    "Peterson", "Gray", "Ramirez", "James", "Watson", "Brooks", "Kelly",
    "Sanders", "Price", "Bennett", "Wood", "Barnes", "Ross", "Henderson",
    "Coleman", "Jenkins", "Perry", "Powell", "Long", "Patterson", "Hughes",
]
EMPLOYEES = [f"EMP{i+1:03d}-{name}" for i, name in enumerate(_SURNAMES)]  # 100 employees

SECURITIES = [
    # Large-cap US equities
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK", "AVGO", "ORCL",
    # Financials
    "JPM", "GS", "MS", "BAC", "WFC", "C", "BLK", "AXP", "SCHW", "USB",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC", "OXY", "HAL",
    # Healthcare
    "PFE", "JNJ", "UNH", "ABBV", "MRK", "BMY", "AMGN", "GILD", "CVS", "CI",
    # Industrials
    "CAT", "DE", "HON", "UPS", "LMT", "RTX", "GE", "MMM", "EMR", "ITW",
    # Technology
    "IBM", "INTC", "AMD", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
]  # 60 securities

COUNTERPARTIES = [
    "Acme Capital LLC", "Brightwater Advisors", "Crestline Partners",
    "DeltaStone Fund", "Eastgate Holdings", "FairPoint Ventures",
]

POLICIES = ["POL-TRADING-2024", "POL-CONFLICT-2024", "POL-BLACKOUT-2024"]


# ---------------------------------------------------------------------------
# Natural language templates for each event type
# ---------------------------------------------------------------------------

NL_TEMPLATES: dict[EventType, list[str]] = {
    EventType.PRE_CLEARANCE_APPROVED: [
        "Pre-clearance request for {ticker} submitted by {emp} was approved on {date}.",
        "Compliance approved {emp}'s request to trade {ticker} (approved {date}).",
        "{emp} received pre-clearance to purchase/sell {ticker} on {date}.",
    ],
    EventType.PRE_CLEARANCE_DENIED: [
        "Pre-clearance for {ticker} requested by {emp} was denied on {date}.",
        "Compliance denied {emp}'s trading request for {ticker} on {date}.",
    ],
    EventType.BLACKOUT_ANNOUNCED: [
        "A trading blackout period for {ticker} was announced effective {date}.",
        "Compliance issued a blackout notice for {ticker} starting {date}.",
        "{ticker} has been placed under a trading blackout as of {date}.",
    ],
    EventType.BLACKOUT_LIFTED: [
        "The blackout period for {ticker} has been lifted as of {date}.",
        "Compliance confirmed the {ticker} blackout is no longer in effect ({date}).",
    ],
    EventType.WATCHLIST_ADDED: [
        "{ticker} was added to the restricted/watch list on {date}.",
        "As of {date}, {ticker} is on the firm's watchlist — all trading requires extra approval.",
    ],
    EventType.WATCHLIST_REMOVED: [
        "{ticker} was removed from the watchlist effective {date}.",
        "Compliance removed {ticker} from restricted list on {date}.",
    ],
    EventType.CONFLICT_DISCLOSED: [
        "{emp} disclosed a potential conflict of interest with {cpty} on {date}.",
        "A conflict-of-interest disclosure was filed by {emp} regarding {cpty} ({date}).",
    ],
    EventType.CONFLICT_AMENDED: [
        "{emp} amended their conflict disclosure for {cpty} on {date}.",
        "An amended COI disclosure from {emp} regarding {cpty} was received {date}.",
    ],
    EventType.CONFLICT_CLEARED: [
        "Compliance cleared the conflict of interest between {emp} and {cpty} on {date}.",
        "The conflict disclosure for {emp}/{cpty} was resolved and cleared on {date}.",
    ],
    EventType.POLICY_ACKNOWLEDGED: [
        "{emp} acknowledged receipt of {pol} on {date}.",
        "Annual policy acknowledgment for {pol} completed by {emp} on {date}.",
    ],
    EventType.POLICY_UPDATED: [
        "Firm policy {pol} was updated on {date}. All prior acknowledgments are now void.",
        "Compliance published a revised version of {pol} on {date} — re-acknowledgment required.",
    ],
    EventType.EMERGENCY_EXCEPTION: [
        "An emergency trading exception was granted to {emp} for {ticker} on {date}.",
        "Compliance approved an exception to the blackout for {emp} re: {ticker} ({date}).",
    ],
    EventType.EXCEPTION_REVOKED: [
        "The emergency exception for {emp} trading {ticker} was revoked on {date}.",
        "Compliance revoked {emp}'s trading exception for {ticker} on {date}.",
    ],
    EventType.TRADE_EXECUTED: [
        "{emp} executed a trade in {ticker} on {date}.",
        "A {ticker} transaction by {emp} was logged on {date}.",
    ],
}


def render_nl(event_type: EventType, emp: str, ticker: Optional[str],
              cpty: Optional[str], pol: Optional[str], date: datetime) -> str:
    template = random.choice(NL_TEMPLATES[event_type])
    date_str = date.strftime("%B %d, %Y")
    return template.format(
        emp=emp, ticker=ticker or "N/A",
        cpty=cpty or "N/A", pol=pol or "N/A", date=date_str
    )


# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------

QUERY_TEMPLATES: dict[QueryType, list[str]] = {
    QueryType.TRADE_COMPLIANT: [
        "Is {emp}'s trade in {ticker} executed on {date} compliant with current policy?",
        "Was {emp} authorized to trade {ticker} on {date} given the current compliance state?",
        "Does {emp}'s {ticker} transaction from {date} pass compliance review?",
    ],
    QueryType.PRECLEARANCE_VALID: [
        "Is the pre-clearance granted to {emp} for {ticker} still valid?",
        "Has {emp}'s pre-clearance for {ticker} been superseded by any subsequent event?",
        "Can {emp} rely on their {ticker} pre-clearance to trade today?",
    ],
    QueryType.CONFLICT_STATUS: [
        "What is the current conflict-of-interest status for {emp} regarding {cpty}?",
        "Is {emp}'s conflict disclosure for {cpty} still the latest filing on record?",
    ],
    QueryType.POLICY_CURRENT: [
        "Has {emp} acknowledged the current version of {pol}?",
        "Is {emp}'s policy acknowledgment for {pol} still valid, or has the policy been updated?",
    ],
    QueryType.AUDIT_TRAIL: [
        "Provide the full provenance chain for {emp}'s compliance status on {ticker}.",
        "What is the audit trail for the pre-clearance decision regarding {emp} and {ticker}?",
    ],
}


def render_query(qtype: QueryType, emp: str, ticker: Optional[str],
                 cpty: Optional[str], pol: Optional[str], date: datetime) -> str:
    template = random.choice(QUERY_TEMPLATES[qtype])
    date_str = date.strftime("%B %d, %Y")
    return template.format(
        emp=emp, ticker=ticker or "N/A",
        cpty=cpty or "N/A", pol=pol or "N/A", date=date_str
    )


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def _base_time(seed: int = 0) -> datetime:
    return datetime(2024, 1, 2, 9, 0, 0) + timedelta(days=seed % 30)


def make_type1_scenario(rng: random.Random) -> dict:
    """
    Type 1 — Simple 2-hop supersession:
      Hop 1: Pre-clearance approved (T)
      Hop 2: Blackout announced (T + δ) → supersedes Hop 1
      Answer: NON_COMPLIANT
    """
    emp = rng.choice(EMPLOYEES)
    ticker = rng.choice(SECURITIES)
    session_id = str(uuid.uuid4())[:8]
    t0 = _base_time(rng.randint(0, 29))

    e1 = Event(
        event_type=EventType.PRE_CLEARANCE_APPROVED,
        timestamp=t0,
        session_id=session_id,
        turn_id=1,
        employee_id=emp,
        security_ticker=ticker,
        raw_text=render_nl(EventType.PRE_CLEARANCE_APPROVED, emp, ticker, None, None, t0),
    )
    t1 = t0 + timedelta(days=rng.randint(1, 3))
    e2 = Event(
        event_type=EventType.BLACKOUT_ANNOUNCED,
        timestamp=t1,
        session_id=session_id,
        turn_id=2,
        employee_id=emp,
        security_ticker=ticker,
        raw_text=render_nl(EventType.BLACKOUT_ANNOUNCED, emp, ticker, None, None, t1),
    )
    kb = KnowledgeBase()
    kb.add(e1)
    kb.add(e2)
    sg = SupersessionGraph(kb=kb)
    sg.build()

    # Ground truth: pre-clearance is superseded, trade is non-compliant
    assert sg.is_superseded(e1), "Type 1: e1 should be superseded"
    assert not sg.is_superseded(e2), "Type 1: e2 (blackout) should be active"

    query = render_query(QueryType.PRECLEARANCE_VALID, emp, ticker, None, None, t1)
    provenance = [
        {"hop": 1, "event_id": e1.event_id, "type": e1.event_type.value,
         "text": e1.raw_text, "status": "superseded"},
        {"hop": 2, "event_id": e2.event_id, "type": e2.event_type.value,
         "text": e2.raw_text, "status": "active", "supersedes": e1.event_id,
         "rule": "R1"},
    ]
    return {
        "example_id": str(uuid.uuid4())[:12],
        "hop_type": "type1_simple_supersession",
        "num_hops": 2,
        "query": query,
        "answer": ComplianceStatus.NON_COMPLIANT.value,
        "answer_reasoning": "Pre-clearance was superseded by subsequent blackout.",
        "provenance": provenance,
        "kb": [{"event_id": e.event_id, "text": e.raw_text,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value}
               for e in kb.events],
        "validity_certificate": sg.validity_certificate(e1),
        "employee": emp,
        "ticker": ticker,
    }


def make_type2_scenario(rng: random.Random) -> dict:
    """
    Type 2 — 3-hop chain supersession:
      Hop 1: Pre-clearance approved
      Hop 2: Blackout announced → supersedes Hop 1
      Hop 3: Emergency exception → supersedes Hop 2
      Answer: COMPLIANT (exception overrides blackout)
    """
    emp = rng.choice(EMPLOYEES)
    ticker = rng.choice(SECURITIES)
    session_id = str(uuid.uuid4())[:8]
    t0 = _base_time(rng.randint(0, 29))

    e1 = Event(
        event_type=EventType.PRE_CLEARANCE_APPROVED,
        timestamp=t0, session_id=session_id, turn_id=1,
        employee_id=emp, security_ticker=ticker,
        raw_text=render_nl(EventType.PRE_CLEARANCE_APPROVED, emp, ticker, None, None, t0),
    )
    t1 = t0 + timedelta(days=rng.randint(1, 2))
    e2 = Event(
        event_type=EventType.BLACKOUT_ANNOUNCED,
        timestamp=t1, session_id=session_id, turn_id=2,
        employee_id=emp, security_ticker=ticker,
        raw_text=render_nl(EventType.BLACKOUT_ANNOUNCED, emp, ticker, None, None, t1),
    )
    t2 = t1 + timedelta(days=rng.randint(1, 2))
    e3 = Event(
        event_type=EventType.EMERGENCY_EXCEPTION,
        timestamp=t2, session_id=session_id, turn_id=3,
        employee_id=emp, security_ticker=ticker,
        raw_text=render_nl(EventType.EMERGENCY_EXCEPTION, emp, ticker, None, None, t2),
    )
    kb = KnowledgeBase()
    for e in [e1, e2, e3]:
        kb.add(e)
    sg = SupersessionGraph(kb=kb)
    sg.build()

    # Ground truth: blackout supersedes pre-clearance, exception supersedes blackout
    assert sg.is_superseded(e1), "Type 2: e1 should be superseded by blackout"
    assert sg.is_superseded(e2), "Type 2: e2 blackout should be superseded by exception"
    assert not sg.is_superseded(e3), "Type 2: e3 exception should be active"

    query = render_query(QueryType.TRADE_COMPLIANT, emp, ticker, None, None, t2)
    provenance = [
        {"hop": 1, "event_id": e1.event_id, "type": e1.event_type.value,
         "text": e1.raw_text, "status": "superseded"},
        {"hop": 2, "event_id": e2.event_id, "type": e2.event_type.value,
         "text": e2.raw_text, "status": "superseded",
         "supersedes": e1.event_id, "rule": "R1"},
        {"hop": 3, "event_id": e3.event_id, "type": e3.event_type.value,
         "text": e3.raw_text, "status": "active",
         "supersedes": e2.event_id, "rule": "R3"},
    ]
    return {
        "example_id": str(uuid.uuid4())[:12],
        "hop_type": "type2_chain_supersession",
        "num_hops": 3,
        "query": query,
        "answer": ComplianceStatus.COMPLIANT.value,
        "answer_reasoning": "Exception overrides blackout; employee is cleared to trade.",
        "provenance": provenance,
        "kb": [{"event_id": e.event_id, "text": e.raw_text,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value}
               for e in kb.events],
        "validity_certificate": sg.validity_certificate(e3),
        "employee": emp,
        "ticker": ticker,
    }


def make_type3_scenario(rng: random.Random) -> dict:
    """
    Type 3 — Cross-session 3-hop supersession:
      Session A: Pre-clearance approved
      Session B (different session, later): Blackout announced
      Session C (another session, later): Blackout lifted
      Answer: COMPLIANT (blackout lifted, original pre-clearance now re-valid)
      Requires joining across 3 sessions.
    """
    emp = rng.choice(EMPLOYEES)
    ticker = rng.choice(SECURITIES)
    sess_a = "sess-" + str(uuid.uuid4())[:6]
    sess_b = "sess-" + str(uuid.uuid4())[:6]
    sess_c = "sess-" + str(uuid.uuid4())[:6]
    t0 = _base_time(rng.randint(0, 20))

    e1 = Event(
        event_type=EventType.PRE_CLEARANCE_APPROVED,
        timestamp=t0, session_id=sess_a, turn_id=3,
        employee_id=emp, security_ticker=ticker,
        raw_text=render_nl(EventType.PRE_CLEARANCE_APPROVED, emp, ticker, None, None, t0),
    )
    t1 = t0 + timedelta(days=rng.randint(2, 4))
    e2 = Event(
        event_type=EventType.BLACKOUT_ANNOUNCED,
        timestamp=t1, session_id=sess_b, turn_id=1,
        employee_id=emp, security_ticker=ticker,
        raw_text=render_nl(EventType.BLACKOUT_ANNOUNCED, emp, ticker, None, None, t1),
    )
    t2 = t1 + timedelta(days=rng.randint(2, 4))
    e3 = Event(
        event_type=EventType.BLACKOUT_LIFTED,
        timestamp=t2, session_id=sess_c, turn_id=2,
        employee_id=emp, security_ticker=ticker,
        raw_text=render_nl(EventType.BLACKOUT_LIFTED, emp, ticker, None, None, t2),
    )
    kb = KnowledgeBase()
    for e in [e1, e2, e3]:
        kb.add(e)
    sg = SupersessionGraph(kb=kb)
    sg.build()

    # Ground truth: blackout supersedes pre-clearance, lift supersedes blackout
    # Note: pre-clearance is transitively superseded, but the LIFT is active.
    # The compliance question is: is the blackout still in effect? No.
    # But is the original pre-clearance still valid? Also no (was superseded by blackout).
    # The employee needs a NEW pre-clearance after the lift.
    assert sg.is_superseded(e2), "Type 3: blackout should be superseded by lift"

    query = render_query(QueryType.PRECLEARANCE_VALID, emp, ticker, None, None, t2)
    provenance = [
        {"hop": 1, "event_id": e1.event_id, "type": e1.event_type.value,
         "text": e1.raw_text, "status": "superseded",
         "session": sess_a},
        {"hop": 2, "event_id": e2.event_id, "type": e2.event_type.value,
         "text": e2.raw_text, "status": "superseded",
         "supersedes": e1.event_id, "rule": "R1", "session": sess_b},
        {"hop": 3, "event_id": e3.event_id, "type": e3.event_type.value,
         "text": e3.raw_text, "status": "active",
         "supersedes": e2.event_id, "rule": "R2", "session": sess_c},
    ]
    # Pre-clearance is void (superseded); blackout is lifted; net state: no active clearance
    return {
        "example_id": str(uuid.uuid4())[:12],
        "hop_type": "type3_cross_session",
        "num_hops": 3,
        "num_sessions": 3,
        "query": query,
        "answer": ComplianceStatus.NON_COMPLIANT.value,
        "answer_reasoning": (
            "Blackout lifted (session C) supersedes the blackout (session B), "
            "but the original pre-clearance (session A) was voided by the blackout. "
            "No valid pre-clearance exists — employee must re-apply."
        ),
        "provenance": provenance,
        "kb": [{"event_id": e.event_id, "text": e.raw_text,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value,
                "session_id": e.session_id}
               for e in kb.events],
        "validity_certificate": sg.validity_certificate(e1),
        "employee": emp,
        "ticker": ticker,
    }


# ---------------------------------------------------------------------------
# Negative examples — for contrast / hard negatives in retrieval eval
# ---------------------------------------------------------------------------

def make_no_supersession_scenario(rng: random.Random) -> dict:
    """
    Baseline: pre-clearance exists, NO blackout ever announced.
    Answer: COMPLIANT. Standard RAG gets this right too.
    Purpose: hard-negative / calibration examples where RAG works.
    """
    emp = rng.choice(EMPLOYEES)
    ticker = rng.choice(SECURITIES)
    session_id = str(uuid.uuid4())[:8]
    t0 = _base_time(rng.randint(0, 29))

    e1 = Event(
        event_type=EventType.PRE_CLEARANCE_APPROVED,
        timestamp=t0, session_id=session_id, turn_id=1,
        employee_id=emp, security_ticker=ticker,
        raw_text=render_nl(EventType.PRE_CLEARANCE_APPROVED, emp, ticker, None, None, t0),
    )
    kb = KnowledgeBase()
    kb.add(e1)
    sg = SupersessionGraph(kb=kb)
    sg.build()

    assert not sg.is_superseded(e1)

    query = render_query(QueryType.PRECLEARANCE_VALID, emp, ticker, None, None, t0)
    return {
        "example_id": str(uuid.uuid4())[:12],
        "hop_type": "type0_no_supersession",
        "num_hops": 1,
        "query": query,
        "answer": ComplianceStatus.ACTIVE.value,
        "answer_reasoning": "Pre-clearance is active; no superseding event found.",
        "provenance": [
            {"hop": 1, "event_id": e1.event_id, "type": e1.event_type.value,
             "text": e1.raw_text, "status": "active"}
        ],
        "kb": [{"event_id": e.event_id, "text": e.raw_text,
                "timestamp": e.timestamp.isoformat(),
                "event_type": e.event_type.value}
               for e in kb.events],
        "validity_certificate": sg.validity_certificate(e1),
        "employee": emp,
        "ticker": ticker,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def _unique_pair(rng: random.Random, used: set) -> tuple[str, str]:
    """Sample a (employee, ticker) pair not yet used in this dataset."""
    for _ in range(10_000):
        emp = rng.choice(EMPLOYEES)
        ticker = rng.choice(SECURITIES)
        key = (emp, ticker)
        if key not in used:
            used.add(key)
            return emp, ticker
    raise RuntimeError(
        f"Entity pool exhausted — increase EMPLOYEES×SECURITIES pool size "
        f"(current: {len(EMPLOYEES)}×{len(SECURITIES)}={len(EMPLOYEES)*len(SECURITIES)}, "
        f"used: {len(used)})"
    )


def _make_with_unique_pair(fn, rng: random.Random, used: set) -> dict:
    """Call a scenario generator but inject a guaranteed-unique (emp, ticker)."""
    emp, ticker = _unique_pair(rng, used)
    # Monkey-patch choices: temporarily make rng.choice return fixed values
    # for the first two calls (emp pool, ticker pool).
    orig_choice = rng.choice
    call_count = [0]
    def patched_choice(seq):
        call_count[0] += 1
        if call_count[0] == 1:
            return emp   # employee pick
        if call_count[0] == 2:
            return ticker  # ticker pick
        return orig_choice(seq)
    rng.choice = patched_choice
    try:
        ex = fn(rng)
    finally:
        rng.choice = orig_choice
    return ex


def generate_dataset(
    n_per_type: int = 250,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> list[dict]:
    """
    Generate a balanced FinSuperQA dataset with UNIQUE (employee, ticker) pairs.

    Uniqueness guarantee: each example has its own (employee_id, ticker) pair,
    so entity+temporal probing of the full corpus will not retrieve events
    that belong to a different query's compliance case. This eliminates
    cross-example contamination in the entity-probe-based retrieval eval.

    n_per_type examples of each hop type:
      - type0 (no supersession): n_per_type
      - type1 (2-hop): n_per_type
      - type2 (3-hop chain): n_per_type
      - type3 (3-hop cross-session): n_per_type
    Total: 4 × n_per_type examples

    Pool capacity: {len(EMPLOYEES)} × {len(SECURITIES)} = {len(EMPLOYEES)*len(SECURITIES)} unique pairs.
    """
    assert n_per_type * 4 <= len(EMPLOYEES) * len(SECURITIES), (
        f"Need {n_per_type*4} unique (emp,ticker) pairs but pool only has "
        f"{len(EMPLOYEES)*len(SECURITIES)}"
    )

    rng = random.Random(seed)
    used_pairs: set = set()
    examples = []

    generators = [
        ("type0", make_no_supersession_scenario),
        ("type1", make_type1_scenario),
        ("type2", make_type2_scenario),
        ("type3", make_type3_scenario),
    ]
    for label, fn in generators:
        for i in range(n_per_type):
            try:
                ex = _make_with_unique_pair(fn, rng, used_pairs)
                examples.append(ex)
            except AssertionError as e:
                print(f"SKIP {label}[{i}]: {e}")
            except RuntimeError as e:
                print(f"ERROR {label}[{i}]: {e}")
                break

    rng.shuffle(examples)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(examples)} examples to {output_path}")

    return examples


if __name__ == "__main__":
    examples = generate_dataset(
        n_per_type=250,
        seed=42,
        output_path="./data/finsuperqa_v1.jsonl",
    )
    # Print summary
    from collections import Counter
    counts = Counter(ex["hop_type"] for ex in examples)
    print("\nDataset summary:")
    for k, v in sorted(counts.items()):
        ans_dist = Counter(ex["answer"] for ex in examples if ex["hop_type"] == k)
        print(f"  {k}: {v} examples | answers: {dict(ans_dist)}")
    print(f"  TOTAL: {len(examples)}")
