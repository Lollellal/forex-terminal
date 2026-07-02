"""Event-Bus-Interface + In-Process-Startimplementierung.

Siehe BACKEND_ARCHITECTURE.md §2.3: Der Domain Layer programmiert nur gegen
das ``EventBus``-Protocol. Eine spätere Broker-basierte Implementierung
(Postgres LISTEN/NOTIFY, Redis Streams, Kafka) kann die In-Process-Variante
ersetzen, ohne dass Aggregates, Repositories oder Subscriber sich ändern
müssen — das ist die konkrete Umsetzung von "neue Module hinzufügen, ohne
bestehende anzupassen".

In Schritt 1 ist noch kein Subscriber verkabelt (kein Risk Monitor, keine
Projection Builder) — das Interface steht bereit für spätere Schritte.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Protocol

from .event_envelope import EventEnvelope

EventHandler = Callable[[EventEnvelope], None]


class EventBus(Protocol):
    def publish(self, events: list[EventEnvelope]) -> None: ...

    def subscribe(self, event_type: str, handler: EventHandler) -> None: ...


class InProcessEventBus:
    """Synchroner Dispatch im selben Prozess — bewusst die einfachste
    Implementierung, die das Interface erfüllt (BACKEND_ARCHITECTURE.md §2.3,
    'Start einfach, Interface zukunftsfest')."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def publish(self, events: list[EventEnvelope]) -> None:
        for event in events:
            for handler in self._handlers.get(event.event_type, []):
                handler(event)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)
