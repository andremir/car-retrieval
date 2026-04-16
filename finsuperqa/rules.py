"""
FinSuperQA Compliance Rule Schema
==================================
Formal specification of the supersession lattice for employee trading compliance.

Each rule is a triple (trigger_event_type, target_event_type, condition) meaning:
  "A trigger event of type T1 supersedes a target event of type T2 if condition holds."

Ground truth derivation is purely rule-based — no human annotation needed.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Event taxonomy
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    # Approvals / clearances
    PRE_CLEARANCE_APPROVED   = "pre_clearance_approved"
    PRE_CLEARANCE_DENIED     = "pre_clearance_denied"
    TRADE_EXECUTED           = "trade_executed"

    # Restrictions
    BLACKOUT_ANNOUNCED       = "blackout_announced"
    BLACKOUT_LIFTED          = "blackout_lifted"
    WATCHLIST_ADDED          = "watchlist_added"
    WATCHLIST_REMOVED        = "watchlist_removed"

    # Disclosures / conflicts
    CONFLICT_DISCLOSED       = "conflict_disclosed"
    CONFLICT_AMENDED         = "conflict_amended"
    CONFLICT_CLEARED         = "conflict_cleared"

    # Policy
    POLICY_ACKNOWLEDGED      = "policy_acknowledged"
    POLICY_UPDATED           = "policy_updated"

    # Exceptions
    EMERGENCY_EXCEPTION      = "emergency_exception"
    EXCEPTION_REVOKED        = "exception_revoked"


# ---------------------------------------------------------------------------
# Supersession rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SupersessionRule:
    """
    Formal rule: trigger_type supersedes target_type on matching entity.
    scope: which entity fields must match between trigger and target.
    label: human-readable description for dataset provenance.
    chain_position: if True, the trigger itself can be superseded further.
    """
    rule_id: str
    trigger_type: EventType
    target_type: EventType
    scope: tuple          # tuple of entity fields that must match
    label: str
    trigger_can_be_superseded: bool = True


# The core supersession lattice (Table 1 in the paper)
SUPERSESSION_RULES: list[SupersessionRule] = [
    # ---- Blackout / pre-clearance ----
    SupersessionRule(
        rule_id="R1",
        trigger_type=EventType.BLACKOUT_ANNOUNCED,
        target_type=EventType.PRE_CLEARANCE_APPROVED,
        scope=("security_ticker",),
        label="Blackout supersedes prior pre-clearance for same security",
    ),
    SupersessionRule(
        rule_id="R2",
        trigger_type=EventType.BLACKOUT_LIFTED,
        target_type=EventType.BLACKOUT_ANNOUNCED,
        scope=("security_ticker",),
        label="Blackout lift supersedes active blackout",
        trigger_can_be_superseded=False,
    ),
    SupersessionRule(
        rule_id="R3",
        trigger_type=EventType.EMERGENCY_EXCEPTION,
        target_type=EventType.BLACKOUT_ANNOUNCED,
        scope=("security_ticker", "employee_id"),
        label="Emergency exception supersedes blackout for specific employee+security",
    ),
    SupersessionRule(
        rule_id="R4",
        trigger_type=EventType.EXCEPTION_REVOKED,
        target_type=EventType.EMERGENCY_EXCEPTION,
        scope=("security_ticker", "employee_id"),
        label="Exception revocation supersedes prior exception",
        trigger_can_be_superseded=False,
    ),

    # ---- Watchlist / approval ----
    SupersessionRule(
        rule_id="R5",
        trigger_type=EventType.WATCHLIST_ADDED,
        target_type=EventType.PRE_CLEARANCE_APPROVED,
        scope=("security_ticker",),
        label="Watchlist addition supersedes all prior approvals for that security",
    ),
    SupersessionRule(
        rule_id="R6",
        trigger_type=EventType.WATCHLIST_REMOVED,
        target_type=EventType.WATCHLIST_ADDED,
        scope=("security_ticker",),
        label="Watchlist removal supersedes the watchlist addition",
        trigger_can_be_superseded=False,
    ),

    # ---- Conflict of interest ----
    SupersessionRule(
        rule_id="R7",
        trigger_type=EventType.CONFLICT_AMENDED,
        target_type=EventType.CONFLICT_DISCLOSED,
        scope=("employee_id", "counterparty"),
        label="Amended disclosure supersedes prior disclosure for same conflict",
    ),
    SupersessionRule(
        rule_id="R8",
        trigger_type=EventType.CONFLICT_CLEARED,
        target_type=EventType.CONFLICT_DISCLOSED,
        scope=("employee_id", "counterparty"),
        label="Conflict cleared supersedes disclosure",
        trigger_can_be_superseded=False,
    ),
    SupersessionRule(
        rule_id="R9",
        trigger_type=EventType.CONFLICT_CLEARED,
        target_type=EventType.CONFLICT_AMENDED,
        scope=("employee_id", "counterparty"),
        label="Conflict cleared supersedes amended disclosure",
        trigger_can_be_superseded=False,
    ),

    # ---- Policy ----
    SupersessionRule(
        rule_id="R10",
        trigger_type=EventType.POLICY_UPDATED,
        target_type=EventType.POLICY_ACKNOWLEDGED,
        scope=("policy_id",),
        label="Policy update voids prior acknowledgment (re-ack required)",
    ),
]

# Quick lookup: for each target type, which triggers supersede it?
SUPERSEDED_BY: dict[EventType, list[SupersessionRule]] = {}
for _r in SUPERSESSION_RULES:
    SUPERSEDED_BY.setdefault(_r.target_type, []).append(_r)


# ---------------------------------------------------------------------------
# Compliance query types
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    TRADE_COMPLIANT       = "trade_compliant"      # Is this trade compliant right now?
    PRECLEARANCE_VALID    = "preclearance_valid"   # Is the pre-clearance still valid?
    CONFLICT_STATUS       = "conflict_status"      # What is the current conflict status?
    POLICY_CURRENT        = "policy_current"       # Has the employee acknowledged the current policy?
    AUDIT_TRAIL           = "audit_trail"          # What is the full provenance chain for this decision?


# ---------------------------------------------------------------------------
# Answer states (ground truth labels)
# ---------------------------------------------------------------------------

class ComplianceStatus(str, Enum):
    COMPLIANT          = "compliant"
    NON_COMPLIANT      = "non_compliant"
    REQUIRES_REVIEW    = "requires_review"
    SUPERSEDED         = "superseded"
    ACTIVE             = "active"
    VOIDED             = "voided"
