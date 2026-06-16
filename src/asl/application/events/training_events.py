"""
TrainingEventBus — Observer pattern.
Decouples the training engine from downstream consumers (logger, MLflow, UI callbacks).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ── Event types ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Phase1CompleteEvent:
    history: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Phase2CompleteEvent:
    history: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingCompleteEvent:
    pass


@dataclass(frozen=True)
class EpochEndEvent:
    epoch: int
    logs: dict[str, float] = field(default_factory=dict)


# ── Event Bus ────────────────────────────────────────────────────────────────

EventType = type
Handler = Callable[[Any], None]


class TrainingEventBus:
    """Simple synchronous publish/subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = {}

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    def publish(self, event: Any) -> None:
        for handler in self._handlers.get(type(event), []):
            try:
                handler(event)
            except Exception:
                # Observer failures must never crash training
                pass
