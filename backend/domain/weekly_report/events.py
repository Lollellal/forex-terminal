"""Event-Typen des WeeklyReport-Aggregates. Siehe BACKEND_ARCHITECTURE.md
§2.2. "Generated" beschreibt hier den Fakt "ein bereits vom Desktop
generierter Report wurde im Backend registriert" — keine Report-Generierung
findet im Backend statt."""

from __future__ import annotations

WEEKLY_REPORT_GENERATED = "WeeklyReportGenerated"
"""Payload: user_id (str), period_start (ISO-Date), period_end (ISO-Date),
content_ref (str)."""

WEEKLY_REPORT_PUBLISHED = "WeeklyReportPublished"
"""Payload: published_at (ISO-String)."""
