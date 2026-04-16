"""
FinSuperQA Supersession Graph
==============================
Directed graph where edge (A → B) means "event A supersedes event B".
Built deterministically from the event KB + supersession rules.

This is the core data structure that makes ground truth derivable without
human annotation: we apply the rules mechanically and read off the state.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from rules import SUPERSESSION_RULES, SupersessionRule, EventType
from events import Event, KnowledgeBase


@dataclass
class SupersessionEdge:
    trigger: Event          # later event that supersedes
    target: Event           # earlier event that is superseded
    rule: SupersessionRule
    is_transitive: bool = False  # True if derived by transitivity


@dataclass
class SupersessionGraph:
    """
    Directed supersession graph over a KnowledgeBase.

    Construction: O(n² × R) where n = |events|, R = |rules|.
    Suitable for KB sizes in the thousands (compliance sessions).
    """
    kb: KnowledgeBase
    edges: list[SupersessionEdge] = field(default_factory=list)
    _superseded_ids: set[str] = field(default_factory=set)   # event_ids that are voided

    def build(self, rules=None) -> None:
        """Populate edges by applying all rules to all ordered event pairs.

        rules: optional list of SupersessionRule objects. Defaults to the
               domain's SUPERSESSION_RULES. Pass a different list to evaluate
               on non-compliance domains (CVE, legal, etc.) without modifying
               the shared module.
        """
        if rules is None:
            rules = SUPERSESSION_RULES
        self._active_rules = rules
        self.edges.clear()
        self._superseded_ids.clear()

        events = self.kb.events  # already sorted by timestamp

        for i, trigger in enumerate(events):
            for j, target in enumerate(events):
                if j >= i:
                    break  # trigger must be STRICTLY later than target
                self._try_supersede(trigger, target)

        # Propagate transitivity:
        # If A supersedes B and B supersedes C, mark C as also superseded by A.
        # We don't add transitive edges to the stored list (keeps graph sparse)
        # but we DO expand _superseded_ids for ground-truth derivation.
        self._propagate_transitive()

    def _try_supersede(self, trigger: Event, target: Event) -> None:
        """Check all rules for trigger→target and add edges if applicable."""
        for rule in getattr(self, "_active_rules", SUPERSESSION_RULES):
            if trigger.event_type != rule.trigger_type:
                continue
            if target.event_type != rule.target_type:
                continue
            # Check entity scope match
            trigger_key = trigger.entity_key(rule.scope)
            target_key = target.entity_key(rule.scope)
            if trigger_key != target_key or None in trigger_key:
                continue
            # Temporal check: trigger must be strictly after target
            if trigger.timestamp <= target.timestamp:
                continue
            edge = SupersessionEdge(trigger=trigger, target=target, rule=rule)
            self.edges.append(edge)
            self._superseded_ids.add(target.event_id)

    def _propagate_transitive(self) -> None:
        """
        Expand supersession through chains.
        E.g. Emergency Exception supersedes Blackout supersedes Pre-clearance
        → Emergency Exception transitively supersedes Pre-clearance.
        We only care about which events are ultimately voided, not new edges.
        """
        # Build adjacency: trigger_id → set of target_ids it directly supersedes
        direct: dict[str, set[str]] = {}
        id_to_event: dict[str, Event] = {e.event_id: e for e in self.kb.events}

        for edge in self.edges:
            direct.setdefault(edge.trigger.event_id, set()).add(edge.target.event_id)

        # BFS from each trigger
        for trigger_id in list(direct.keys()):
            visited = set()
            queue = list(direct.get(trigger_id, []))
            while queue:
                tid = queue.pop()
                if tid in visited:
                    continue
                visited.add(tid)
                self._superseded_ids.add(tid)
                # If this target is also a trigger of something, follow chain
                for next_tid in direct.get(tid, []):
                    queue.append(next_tid)

    def is_superseded(self, event: Event) -> bool:
        return event.event_id in self._superseded_ids

    def active_events(self) -> list[Event]:
        """Return all events that have not been superseded."""
        return [e for e in self.kb.events if not self.is_superseded(e)]

    def superseding_chain(self, event: Event) -> list[SupersessionEdge]:
        """
        Return the ordered chain of edges that supersede this event.
        Used to build the provenance chain P = [e1, e2, ..., ek].
        """
        chain = []
        current_id = event.event_id
        seen = set()
        while current_id not in seen:
            seen.add(current_id)
            for edge in self.edges:
                if edge.target.event_id == current_id:
                    chain.append(edge)
                    current_id = edge.trigger.event_id
                    break
            else:
                break
        return chain

    def validity_certificate(self, event: Event) -> dict:
        """
        Return a certificate confirming that no event in the KB supersedes `event`.
        Used as the "no superseding evidence" check in TCA metric.
        """
        if self.is_superseded(event):
            chain = self.superseding_chain(event)
            return {
                "valid": False,
                "reason": "superseded",
                "superseding_chain": [
                    {"event_id": e.trigger.event_id, "rule": e.rule.rule_id}
                    for e in chain
                ],
            }
        return {
            "valid": True,
            "reason": "no_superseding_event_found",
            "superseding_chain": [],
        }
