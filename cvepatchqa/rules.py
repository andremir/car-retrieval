"""
CVE/Security-Patch Domain — Supersession Rules
================================================
Entity mapping onto the shared Event dataclass:
  employee_id      → product_id   (e.g. "nginx-1.24", "openssl-3.0")
  security_ticker  → cve_id       (e.g. "CVE-2024-00042")

Supersession rules:
  R_CVE1: PATCH_RELEASED   supersedes CVE_DISCLOSED      (scope: cve_id)
  R_CVE2: PATCH_SUPERSEDED supersedes PATCH_RELEASED     (scope: cve_id)
  R_CVE3: CVE_REOPENED     supersedes PATCH_RELEASED     (scope: cve_id)
  R_CVE4: WORKAROUND_ISSUED supersedes CVE_DISCLOSED     (scope: cve_id, product_id)

Vocabulary gap design:
  CVE_DISCLOSED  → "actively exploited", "attack vector", "no patch", "severity"
  PATCH_RELEASED → "security update", "resolved in version", "apply patch", "mitigate"
  Queries        → "still affected", "active vulnerability", "security flaw"

This guarantees BM25/Dense retrieve the disclosure (query vocabulary match) but
miss the patch (disjoint vocabulary), producing TCA = 0.000 on T1.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):
    CVE_DISCLOSED      = "cve_disclosed"
    PATCH_RELEASED     = "patch_released"
    PATCH_SUPERSEDED   = "patch_superseded"   # Newer patch supersedes prior patch
    WORKAROUND_ISSUED  = "workaround_issued"
    CVE_REOPENED       = "cve_reopened"       # Patch found insufficient


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
        rule_id="R_CVE1",
        trigger_type=EventType.PATCH_RELEASED,
        target_type=EventType.CVE_DISCLOSED,
        scope=("security_ticker",),   # security_ticker = cve_id
        label="Patch supersedes vulnerability disclosure for same CVE",
    ),
    SupersessionRule(
        rule_id="R_CVE2",
        trigger_type=EventType.PATCH_SUPERSEDED,
        target_type=EventType.PATCH_RELEASED,
        scope=("security_ticker",),
        label="Updated patch supersedes prior patch for same CVE",
        trigger_can_be_superseded=False,
    ),
    SupersessionRule(
        rule_id="R_CVE3",
        trigger_type=EventType.CVE_REOPENED,
        target_type=EventType.PATCH_RELEASED,
        scope=("security_ticker",),
        label="CVE reopened (patch insufficient) supersedes prior patch",
        trigger_can_be_superseded=True,
    ),
    SupersessionRule(
        rule_id="R_CVE4",
        trigger_type=EventType.WORKAROUND_ISSUED,
        target_type=EventType.CVE_DISCLOSED,
        scope=("security_ticker", "employee_id"),   # employee_id = product_id
        label="Workaround supersedes raw disclosure for same (CVE, product)",
        trigger_can_be_superseded=True,
    ),
]

SUPERSEDED_BY: dict[EventType, list[SupersessionRule]] = {}
for _r in SUPERSESSION_RULES:
    SUPERSEDED_BY.setdefault(_r.target_type, []).append(_r)
