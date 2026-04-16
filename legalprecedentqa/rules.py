"""
Legal-Precedent Domain — Supersession Rules
============================================
Entity mapping onto the shared Event dataclass:
  employee_id     → case_slug    (e.g. "roe_v_wade_410us113")
  security_ticker → legal_qid    (e.g. "LEGALQ-007")

Supersession rules:
  R_LEG1: RULING_OVERRULED   supersedes RULING_ISSUED      (scope: case_slug)
  R_LEG2: RULING_CODIFIED    supersedes RULING_ISSUED      (scope: legal_qid)
  R_LEG3: RULING_OVERRULED   supersedes RULING_CODIFIED    (scope: legal_qid)

Vocabulary gap design:
  RULING_ISSUED    → "held that", "controlling precedent", "the standard is"
  RULING_OVERRULED → "overruled", "abrogated", "no longer controlling"
  Queries          → "still good law", "controlling authority", "valid holding"

BM25/Dense retrieve the original ruling (high query-vocabulary overlap) but
miss the overruling notice (disjoint vocabulary), producing TCA = 0.000 on T1.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):
    RULING_ISSUED      = "ruling_issued"
    RULING_OVERRULED   = "ruling_overruled"
    RULING_CODIFIED    = "ruling_codified"    # Statute supersedes common-law rule
    RULING_AFFIRMED    = "ruling_affirmed"    # Later court affirms ruling (T0 context)


@dataclass(frozen=True)
class SupersessionRule:
    rule_id: str
    trigger_type: EventType
    target_type: EventType
    scope: tuple
    label: str
    trigger_can_be_superseded: bool = True


SUPERSESSION_RULES: list[SupersessionRule] = [
    SupersessionRule(
        rule_id="R_LEG1",
        trigger_type=EventType.RULING_OVERRULED,
        target_type=EventType.RULING_ISSUED,
        scope=("employee_id",),   # employee_id = case_slug
        label="Overruling decision voids prior ruling for same case citation",
        trigger_can_be_superseded=False,
    ),
    SupersessionRule(
        rule_id="R_LEG2",
        trigger_type=EventType.RULING_CODIFIED,
        target_type=EventType.RULING_ISSUED,
        scope=("security_ticker",),   # security_ticker = legal_qid
        label="Statutory codification supersedes common-law ruling on same question",
        trigger_can_be_superseded=False,
    ),
    SupersessionRule(
        rule_id="R_LEG3",
        trigger_type=EventType.RULING_OVERRULED,
        target_type=EventType.RULING_CODIFIED,
        scope=("security_ticker",),
        label="Constitutional overruling supersedes statutory rule on same question",
        trigger_can_be_superseded=False,
    ),
]

SUPERSEDED_BY: dict[EventType, list[SupersessionRule]] = {}
for _r in SUPERSESSION_RULES:
    SUPERSEDED_BY.setdefault(_r.target_type, []).append(_r)
