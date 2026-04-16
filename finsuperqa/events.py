"""
FinSuperQA Event Data Model
============================
Typed event objects that populate the knowledge base.
Every event has: type, timestamp, entity fields, session_id, turn_id.
The supersession graph is derived deterministically from events + rules.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from rules import EventType


@dataclass
class Event:
    """
    A single compliance event extracted from a conversation turn.

    Fields:
        event_id     : unique identifier
        event_type   : EventType
        timestamp    : when the event occurred (used for temporal ordering)
        session_id   : which conversation session this came from
        turn_id      : turn index within session (for multi-session joins)
        employee_id  : the employee referenced
        security_ticker : the security referenced (if any)
        counterparty : for conflict-of-interest events
        policy_id    : for policy events
        raw_text     : the natural language sentence expressing this event
        metadata     : any additional key-value pairs
    """
    event_type: EventType
    timestamp: datetime
    session_id: str
    turn_id: int
    employee_id: str
    raw_text: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    security_ticker: Optional[str] = None
    counterparty: Optional[str] = None
    policy_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def entity_key(self, scope: tuple) -> tuple:
        """Return the entity values for the given scope fields."""
        vals = []
        for field_name in scope:
            vals.append(getattr(self, field_name, None))
        return tuple(vals)

    def __repr__(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M")
        return (
            f"Event({self.event_type.value}, "
            f"emp={self.employee_id}, "
            f"ticker={self.security_ticker}, "
            f"t={ts}, "
            f"sess={self.session_id}:{self.turn_id})"
        )


@dataclass
class KnowledgeBase:
    """
    A temporally-ordered sequence of events (the RAG corpus).
    Events are stored in timestamp order.
    The supersession graph is built on demand.
    """
    events: list[Event] = field(default_factory=list)

    def add(self, event: Event) -> None:
        self.events.append(event)
        self.events.sort(key=lambda e: (e.timestamp, e.session_id, e.turn_id))

    def events_by_type(self, event_type: EventType) -> list[Event]:
        return [e for e in self.events if e.event_type == event_type]

    def events_before(self, timestamp: datetime) -> list[Event]:
        return [e for e in self.events if e.timestamp < timestamp]

    def __len__(self) -> int:
        return len(self.events)
