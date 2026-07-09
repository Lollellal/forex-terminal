"""
Historischer Backfill fuer den Wirtschaftskalender (investing.com).

Laedt den Kalender monatsweise fuer einen Datumsbereich, mit:
  - Checkpoint-Datei (JSON): welche Monate bereits erfolgreich geladen wurden
  - Resume: bei Neustart werden abgeschlossene Monate uebersprungen
  - Rate-Limit: fixe Pause zwischen Requests, um Bot-Schutz zu vermeiden
  - Batches: max. BATCH_SIZE Monate pro Batch, mit langer Pause zwischen Batches
  - Bot-Block-Erkennung (HTTP 403/429): stoppt den Batch sofort sauber, statt
    weitere Monate faelschlich als "failed" (= keine Daten) zu markieren
  - Logging: Fortschritt pro Monat

Jeder Monat wird als eigene Parquet-Datei gespeichert (data/raw/calendar/calendar_g7_YYYYMM.parquet),
damit ein Abbruch keine bereits gespeicherten Monate gefaehrdet.

Bekannt aus Live-Test (2026-07-09): investing.com blockt nach ca. 25-27 Requests
in Folge mit HTTP 429, auch bei 5s Pause zwischen Requests. Gemessener Cooldown
bis der Block wieder aufgehoben war: ~15 Minuten. Daher 20s Rate-Limit,
Batches von max. 20 Monaten, 10-15 Min Pause zwischen Batches.

Nutzung:
    python -m src.data.calendar_backfill --start 2015-01 --end 2015-12
    python -m src.data.calendar_backfill --start 2012-01 --end 2026-06 --batched
"""

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.calendar_fetcher import (
    BotBlockedError,
    fetch_raw_calendar,
    filter_g7_currencies,
    filter_high_impact,
    parse_calendar_html,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw/calendar")
CHECKPOINT_PATH = Path("data/raw/calendar/backfill_checkpoint.json")

# Pause zwischen zwei Monats-Requests (Sekunden) — schont die API, vermeidet Bot-Block
RATE_LIMIT_SECONDS = 20.0

# Batch-Steuerung fuer lange Backfills (siehe Docstring oben)
BATCH_SIZE_MONTHS = 20
BATCH_PAUSE_SECONDS = 750.0   # 12.5 Min — Mitte des gemessenen 10-15-Min-Cooldown-Fensters


# ── Monats-Iteration ─────────────────────────────────────────────────────────

def _month_range(start: str, end: str) -> list[str]:
    """Erzeugt eine Liste von 'YYYY-MM'-Strings von start bis end (inklusiv)."""
    start_date = date.fromisoformat(f"{start}-01")
    end_date = date.fromisoformat(f"{end}-01")

    months = []
    current = start_date
    while current <= end_date:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)
    return months


def _month_bounds(month: str) -> tuple[str, str]:
    """'2015-06' → ('2015-06-01', '2015-06-30')."""
    year, mon = (int(x) for x in month.split("-"))
    start = date(year, mon, 1)
    if mon == 12:
        end = date(year, 12, 31)
    else:
        next_month = date(year, mon + 1, 1)
        end = date(next_month.year, next_month.month, next_month.day)
        end = date.fromordinal(end.toordinal() - 1)
    return start.isoformat(), end.isoformat()


# ── Checkpoint ────────────────────────────────────────────────────────────

def _load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        return {"done": [], "failed": []}
    with open(checkpoint_path, encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(checkpoint_path: Path, checkpoint: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


# ── Monats-Fetch ──────────────────────────────────────────────────────────

def _month_out_path(month: str, out_dir: Path) -> Path:
    return out_dir / f"calendar_g7_{month.replace('-', '')}.parquet"


def fetch_month(month: str, strict_filter: bool = False) -> pd.DataFrame:
    """Laedt und parst einen einzelnen Monat. Wirft bei leerem HTML (echtes
    "keine Daten", z.B. Timeout/Connection-Fehler) keine Exception, sondern
    gibt einen leeren DataFrame zurueck. Bei Bot-Block (HTTP 403/429) wird
    BotBlockedError NICHT abgefangen, sondern an den Aufrufer durchgereicht —
    das ist ein anderer Fehlerfall als "keine Daten" und muss den Batch stoppen."""
    date_from, date_to = _month_bounds(month)
    html = fetch_raw_calendar(date_from=date_from, date_to=date_to)
    if not html:
        return pd.DataFrame()

    df = parse_calendar_html(html)
    df = filter_g7_currencies(df)
    df = filter_high_impact(df, strict=strict_filter)
    return df


# ── Backfill-Runner ───────────────────────────────────────────────────────

def run_backfill(
    start: str,
    end: str,
    out_dir: Path = DATA_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    rate_limit_seconds: float = RATE_LIMIT_SECONDS,
    strict_filter: bool = False,
) -> dict:
    """
    Fuehrt den Backfill fuer alle Monate von start bis end ('YYYY-MM') aus.

    Bereits in der Checkpoint-Datei als 'done' markierte Monate werden uebersprungen
    (Resume nach Abbruch). Fehlgeschlagene Monate (echtes "keine Daten", z.B.
    Timeout) landen unter 'failed' und werden beim naechsten Lauf erneut versucht.

    Bei Bot-Block (HTTP 403/429) wird der Batch SOFORT sauber gestoppt — der
    blockierte Monat wird NICHT als 'failed' markiert (er hat schlicht noch
    keinen Versuch bekommen), sondern bleibt fuer den naechsten Lauf offen.
    Weitere Requests waehrend eines aktiven Blocks wuerden ihn nur verlaengern.

    Returns: Zusammenfassung {"done": [...], "failed": [...], "skipped": [...],
             "blocked_at": <Monat oder None>}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = _load_checkpoint(checkpoint_path)
    months = _month_range(start, end)

    skipped: list[str] = []
    newly_done: list[str] = []
    newly_failed: list[str] = []
    blocked_at: str | None = None

    for i, month in enumerate(months):
        if month in checkpoint["done"]:
            logger.info("Monat %s bereits abgeschlossen — ueberspringe", month)
            skipped.append(month)
            continue

        logger.info("Backfill Monat %s (%d/%d) ...", month, i + 1, len(months))
        try:
            df = fetch_month(month, strict_filter=strict_filter)
        except BotBlockedError as exc:
            logger.error(
                "Bot-Block bei Monat %s erkannt — stoppe Batch sofort (%s)", month, exc
            )
            blocked_at = month
            break
        except Exception as exc:
            logger.error("Unerwarteter Fehler bei Monat %s: %s", month, exc)
            df = pd.DataFrame()

        if df.empty:
            logger.warning("Monat %s: keine Daten erhalten — als 'failed' markiert", month)
            if month not in checkpoint["failed"]:
                checkpoint["failed"].append(month)
            newly_failed.append(month)
        else:
            out_path = _month_out_path(month, out_dir)
            df.to_parquet(out_path, index=False)
            logger.info("Monat %s: %d Events gespeichert → %s", month, len(df), out_path)
            checkpoint["done"].append(month)
            if month in checkpoint["failed"]:
                checkpoint["failed"].remove(month)
            newly_done.append(month)

        # Checkpoint nach jedem Monat sichern — Fortschritt geht bei Abbruch nicht verloren
        _save_checkpoint(checkpoint_path, checkpoint)

        if i < len(months) - 1:
            time.sleep(rate_limit_seconds)

    logger.info(
        "Backfill-Batch beendet: %d neu, %d uebersprungen, %d fehlgeschlagen%s",
        len(newly_done), len(skipped), len(newly_failed),
        f", blockiert bei {blocked_at}" if blocked_at else "",
    )
    return {
        "done": newly_done,
        "skipped": skipped,
        "failed": newly_failed,
        "blocked_at": blocked_at,
    }


# ── Batch-Runner ──────────────────────────────────────────────────────────

def _chunk_months(months: list[str], batch_size: int) -> list[list[str]]:
    return [months[i:i + batch_size] for i in range(0, len(months), batch_size)]


def run_backfill_batched(
    start: str,
    end: str,
    out_dir: Path = DATA_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    rate_limit_seconds: float = RATE_LIMIT_SECONDS,
    batch_size: int = BATCH_SIZE_MONTHS,
    batch_pause_seconds: float = BATCH_PAUSE_SECONDS,
    strict_filter: bool = False,
) -> dict:
    """
    Fuehrt den Backfill in Batches von max. `batch_size` Monaten aus, mit
    `batch_pause_seconds` Pause zwischen den Batches.

    Wird ein Bot-Block (HTTP 403/429) erkannt, stoppt der GESAMTE Lauf sofort
    (nicht nur der aktuelle Batch) — waehrend eines aktiven Blocks wuerden
    weitere Batches ihn nur verlaengern. Der Fortschritt bis dahin bleibt im
    Checkpoint erhalten; ein erneuter Aufruf setzt automatisch dort fort, wo
    aufgehoert wurde.

    Returns: Zusammenfassung {"done": [...], "failed": [...], "skipped": [...],
             "blocked_at": <Monat oder None>, "batches_run": <int>}
    """
    months = _month_range(start, end)
    batches = _chunk_months(months, batch_size)

    all_done: list[str] = []
    all_skipped: list[str] = []
    all_failed: list[str] = []
    blocked_at: str | None = None
    batches_run = 0

    for batch_idx, batch_months in enumerate(batches):
        logger.info(
            "── Batch %d/%d: %s bis %s ──",
            batch_idx + 1, len(batches), batch_months[0], batch_months[-1],
        )
        summary = run_backfill(
            start=batch_months[0],
            end=batch_months[-1],
            out_dir=out_dir,
            checkpoint_path=checkpoint_path,
            rate_limit_seconds=rate_limit_seconds,
            strict_filter=strict_filter,
        )
        batches_run += 1
        all_done.extend(summary["done"])
        all_skipped.extend(summary["skipped"])
        all_failed.extend(summary["failed"])

        if summary["blocked_at"]:
            blocked_at = summary["blocked_at"]
            logger.error(
                "Bot-Block in Batch %d/%d — gesamter Lauf gestoppt. "
                "Checkpoint ist aktuell, spaeterer Resume laedt automatisch weiter.",
                batch_idx + 1, len(batches),
            )
            break

        if batch_idx < len(batches) - 1:
            logger.info("Batch-Pause: %.0f s ...", batch_pause_seconds)
            time.sleep(batch_pause_seconds)

    logger.info(
        "Backfill (batched) beendet: %d Batches, %d neu, %d uebersprungen, %d fehlgeschlagen%s",
        batches_run, len(all_done), len(all_skipped), len(all_failed),
        f", blockiert bei {blocked_at}" if blocked_at else "",
    )
    return {
        "done": all_done,
        "skipped": all_skipped,
        "failed": all_failed,
        "blocked_at": blocked_at,
        "batches_run": batches_run,
    }


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Historischer Kalender-Backfill")
    parser.add_argument("--start", required=True, help="Start-Monat, z.B. 2015-01")
    parser.add_argument("--end", required=True, help="End-Monat, z.B. 2015-12")
    parser.add_argument("--out-dir", default=str(DATA_DIR))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--rate-limit", type=float, default=RATE_LIMIT_SECONDS)
    parser.add_argument("--batched", action="store_true",
                         help="In Batches mit Pausen fahren (empfohlen fuer lange Zeitraeume)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_MONTHS)
    parser.add_argument("--batch-pause", type=float, default=BATCH_PAUSE_SECONDS)
    args = parser.parse_args()

    if args.batched:
        summary = run_backfill_batched(
            start=args.start,
            end=args.end,
            out_dir=Path(args.out_dir),
            checkpoint_path=Path(args.checkpoint),
            rate_limit_seconds=args.rate_limit,
            batch_size=args.batch_size,
            batch_pause_seconds=args.batch_pause,
        )
    else:
        summary = run_backfill(
            start=args.start,
            end=args.end,
            out_dir=Path(args.out_dir),
            checkpoint_path=Path(args.checkpoint),
            rate_limit_seconds=args.rate_limit,
        )
    print(summary)
