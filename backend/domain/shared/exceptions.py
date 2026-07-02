"""Fehlerklassen des Shared Kernel."""


class ConcurrencyConflictError(Exception):
    """Zwei konkurrierende Commands haben versucht, dieselbe Aggregate-Version
    zu schreiben. Wird durch den UNIQUE-Constraint auf
    core.event_store (aggregate_type, aggregate_id, version) erzwungen —
    siehe BACKEND_ARCHITECTURE.md §2.1."""


class EventStreamGapError(Exception):
    """Der geladene Event-Stream eines Aggregates hat eine Versionslücke.
    Darf unter normalem Betrieb nie auftreten (append() ist atomar und
    versionslückenlos) — ein Auftreten deutet auf Datenkorruption oder
    einen Bug im Persistenzpfad hin."""
