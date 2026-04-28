"""
Wirtschaftskalender-Fetcher fuer G7-Laender.

Scraped den Investing.com Economic Calendar ueber deren interne AJAX-API,
filtert High-Impact Events (NFP, CPI, FOMC, EZB etc.) und speichert als Parquet.

Endpoint:  POST https://www.investing.com/economic-calendar/Service/getCalendarFilteredData
Antwort:   JSON { "data": "<HTML-Fragment>", "rows_num": N, ... }
Das HTML-Fragment wird mit BeautifulSoup geparst.

Robustheit:
  - Retry mit exponentiellem Backoff bei Timeouts / Verbindungsfehlern
  - Zwei-Stufen-Filter: Investing.com 3-Sterne-Rating + Keyword-Whitelist
  - Drei Fallback-Methoden fuer Impact-Extraktion aus unterschiedlichen HTML-Varianten
  - Bei vollstaendigem Fehlschlag: leerer DataFrame statt Exception

Output: Parquet in data/raw/calendar/
"""

import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Konstanten ─────────────────────────────────────────────────────────────

CALENDAR_URL = (
    "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
)
CALENDAR_REFERER = "https://www.investing.com/economic-calendar/"

DATA_DIR = Path("data/raw/calendar")

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0       # Sekunden; verdoppelt sich je Versuch
BOT_BLOCK_CODES = {403, 429}

DEFAULT_LOOKAHEAD_DAYS = 14

# investing.com Country-IDs fuer G7-Waehrungen
G7_COUNTRY_IDS: dict[str, int] = {
    "USD": 5,
    "EUR": 72,
    "GBP": 4,
    "JPY": 35,
    "CAD": 6,
    "CHF": 12,
    "AUD": 25,
}

COUNTRY_ID_TO_CURRENCY: dict[int, str] = {v: k for k, v in G7_COUNTRY_IDS.items()}

# Bekannte High-Impact Stichworte als Fallback-Filter (Kleinbuchstaben)
HIGH_IMPACT_KEYWORDS: frozenset[str] = frozenset({
    "nonfarm payroll", "nfp",
    "cpi", "consumer price index", "core cpi", "core pce",
    "fomc", "federal funds rate", "federal open market",
    "ecb", "european central bank",
    "boe", "bank of england",
    "boj", "bank of japan",
    "rba", "reserve bank of australia",
    "boc", "bank of canada",
    "snb", "swiss national bank",
    "interest rate decision",
    "gdp", "gross domestic product",
    "unemployment rate",
    "retail sales",
    "ism manufacturing", "ism services",
    "pmi",
    "average hourly earnings",
    "adp employment",
    "initial jobless claims",
    "trade balance",
    "inflation",
})

# ── HTTP-Request ──────────────────────────────────────────────────────────

def _make_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "Referer": CALENDAR_REFERER,
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _build_form_data(
    date_from: str,
    date_to: str,
    country_ids: list[int],
) -> dict:
    """POST-Body fuer die investing.com Kalender-AJAX-API."""
    data: dict = {
        "timeZone": "55",          # UTC
        "timeFilter": "timeOnly",
        "currentTab": "custom",
        "submitFilters": "1",
        "limit_from": "0",
        "dateFrom": date_from,
        "dateTo": date_to,
    }
    # Mehrfach-Werte: country[] und importance[] als Listen
    data["country[]"] = [str(cid) for cid in country_ids]
    data["importance[]"] = ["3"]   # 3 = high impact
    return data


def fetch_raw_calendar(
    date_from: str | None = None,
    date_to: str | None = None,
    country_ids: list[int] | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Laedt den Rohkalender von investing.com als HTML-Fragment-String.

    Gibt leeren String zurueck wenn alle Retries fehlschlagen —
    niemals Exception nach aussen.
    """
    today = datetime.now(timezone.utc)
    if date_from is None:
        date_from = today.strftime("%Y-%m-%d")
    if date_to is None:
        date_to = (today + timedelta(days=DEFAULT_LOOKAHEAD_DAYS)).strftime("%Y-%m-%d")
    if country_ids is None:
        country_ids = list(G7_COUNTRY_IDS.values())

    form_data = _build_form_data(date_from, date_to, country_ids)
    headers = _make_headers()
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Lade Kalender %s – %s (Versuch %d/%d) ...",
                date_from, date_to, attempt, max_retries,
            )
            resp = requests.post(
                CALENDAR_URL,
                headers=headers,
                data=form_data,
                timeout=30,
            )
            resp.raise_for_status()

            payload = resp.json()
            html_fragment = payload.get("data", "")
            if not html_fragment:
                logger.warning("Leere Antwort von investing.com (Versuch %d)", attempt)
            else:
                logger.info("HTML-Fragment empfangen (%d Bytes)", len(html_fragment))
                return html_fragment

        except requests.exceptions.Timeout as exc:
            logger.warning("Timeout (Versuch %d): %s", attempt, exc)
            last_error = exc

        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP-Fehler (Versuch %d): %s", attempt, exc)
            last_error = exc
            if exc.response is not None and exc.response.status_code in BOT_BLOCK_CODES:
                wait = RETRY_BACKOFF ** attempt * 3
                logger.warning(
                    "Bot-Schutz erkannt (HTTP %d) — warte %.0f s ...",
                    exc.response.status_code, wait,
                )
                time.sleep(wait)
                continue

        except (requests.exceptions.ConnectionError, ValueError) as exc:
            logger.warning("Verbindungsfehler (Versuch %d): %s", attempt, exc)
            last_error = exc

        if attempt < max_retries:
            wait = RETRY_BACKOFF ** attempt
            logger.info("Warte %.0f s vor naechstem Versuch ...", wait)
            time.sleep(wait)

    logger.error("Alle Retries fehlgeschlagen. Letzter Fehler: %s", last_error)
    return ""


# ── HTML-Parsing ──────────────────────────────────────────────────────────

def _parse_impact(row: BeautifulSoup) -> str:
    """
    Extrahiert den Impact-Level aus einer Event-Tabellenzeile.

    Methode 1: class-Attribut der Zeile (z.B. 'impact-3')
    Methode 2: Anzahl grayFullBullishIcon-Icons im sentiment-td
    Methode 3: title/aria-label-Text der td-Elemente
    """
    row_classes = " ".join(row.get("class", []))

    # Methode 1: Explizite Impact-Klasse auf der Zeile
    for level, name in [(3, "high"), (2, "medium"), (1, "low")]:
        if f"impact-{level}" in row_classes:
            return name

    # Methode 2: Icon-Zaehlung im sentiment-td
    sentiment_td = row.find("td", class_="sentiment")
    if sentiment_td:
        icons = sentiment_td.find_all("i", class_="grayFullBullishIcon")
        n = len(icons)
        if n >= 3:
            return "high"
        if n == 2:
            return "medium"
        if n == 1:
            return "low"

    # Methode 3: title/aria-label an beliebigen td-Elementen
    for td in row.find_all("td"):
        hint = (td.get("title", "") or td.get("aria-label", "")).lower()
        if "high" in hint:
            return "high"
        if "medium" in hint:
            return "medium"
        if "low" in hint:
            return "low"

    return "unknown"


def _parse_currency(row: BeautifulSoup) -> str:
    """
    Extrahiert den Waehrungscode aus einer Event-Tabellenzeile.

    Methode 1: 3-Buchstaben-Code aus flagCur-td (z.B. ' USD')
    Methode 2: data-country_id-Attribut der Zeile
    """
    flag_td = row.find("td", class_="flagCur")
    if flag_td:
        text = flag_td.get_text(separator=" ", strip=True)
        match = re.search(r"\b([A-Z]{3})\b", text)
        if match:
            return match.group(1)

    country_id_str = row.get("data-country_id")
    if country_id_str:
        try:
            cid = int(country_id_str)
            currency = COUNTRY_ID_TO_CURRENCY.get(cid)
            if currency:
                return currency
        except (ValueError, TypeError):
            pass

    return "UNKNOWN"


def _normalize_date(text: str) -> str:
    """Konvertiert Datumstexte in YYYY-MM-DD (mehrere Formate werden probiert)."""
    text = text.strip()
    for fmt in (
        "%A, %B %d, %Y",   # Monday, April 28, 2026
        "%B %d, %Y",        # April 28, 2026
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    logger.debug("Datum konnte nicht normalisiert werden: '%s'", text)
    return text


def _empty_calendar_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "time", "currency", "event_name", "impact", "forecast", "previous"]
    )


def parse_calendar_html(html: str) -> pd.DataFrame:
    """
    Parst das HTML-Fragment der investing.com Kalender-API.

    Erkennt Datums-Trennzeilen (class='theDay') und Event-Zeilen
    (class='js-event-item'). Gibt leeren DataFrame bei leerem Input zurueck.
    """
    if not html or not html.strip():
        logger.warning("Leeres HTML — gebe leeren DataFrame zurueck")
        return _empty_calendar_df()

    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []
    current_date: str = ""

    for row in soup.find_all("tr"):
        classes = row.get("class", [])

        # Datums-Trennzeile
        if "theDay" in classes:
            current_date = _normalize_date(row.get_text(separator=" ", strip=True))
            continue

        # Nur Event-Zeilen verarbeiten
        if "js-event-item" not in classes:
            continue

        try:
            time_td = row.find("td", class_="time")
            event_time = time_td.get_text(strip=True) if time_td else ""

            currency = _parse_currency(row)
            impact = _parse_impact(row)

            event_td = row.find("td", class_="event")
            if event_td is None:
                continue
            event_name = event_td.get_text(strip=True)
            if not event_name:
                continue

            # Forecast und Previous via ID-Konvention: eventRowId_X → eventForecast_X
            row_id = row.get("id", "")
            forecast_id = row_id.replace("eventRowId_", "eventForecast_")
            previous_id = row_id.replace("eventRowId_", "eventPrevious_")

            forecast_td = soup.find("td", id=forecast_id) if forecast_id else None
            previous_td = soup.find("td", id=previous_id) if previous_id else None

            records.append({
                "date": current_date,
                "time": event_time,
                "currency": currency,
                "event_name": event_name,
                "impact": impact,
                "forecast": forecast_td.get_text(strip=True) if forecast_td else "",
                "previous": previous_td.get_text(strip=True) if previous_td else "",
            })

        except Exception as exc:
            logger.debug("Zeile konnte nicht geparst werden: %s", exc)
            continue

    if not records:
        logger.warning("Keine Events aus HTML extrahiert")
        return _empty_calendar_df()

    df = pd.DataFrame(records)
    logger.info("Geparste Events: %d", len(df))
    return df


# ── Filter ────────────────────────────────────────────────────────────────

def filter_high_impact(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    Behaelt nur High-Impact Events.

    strict=False (Standard): impact=='high' ODER Keyword-Match (Fallback fuer
                             Events, bei denen der Impact-Level nicht ermittelt
                             werden konnte).
    strict=True: ausschliesslich impact=='high'.
    """
    if df.empty:
        return df

    is_high = df["impact"] == "high"

    if strict:
        result = df[is_high].reset_index(drop=True)
    else:
        def _keyword_match(name: str) -> bool:
            lower = name.lower()
            return any(kw in lower for kw in HIGH_IMPACT_KEYWORDS)

        is_keyword = df["event_name"].apply(_keyword_match)
        result = df[is_high | is_keyword].copy().reset_index(drop=True)

    logger.info(
        "Nach High-Impact-Filter: %d Events (von %d gesamt)", len(result), len(df)
    )
    return result


def filter_g7_currencies(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt Events fuer Laender ausserhalb der G7-Waehrungen."""
    if df.empty:
        return df
    g7 = set(G7_COUNTRY_IDS.keys())
    return df[df["currency"].isin(g7)].reset_index(drop=True)


# ── Speichern ─────────────────────────────────────────────────────────────

def save_to_parquet(df: pd.DataFrame, out_dir: Path = DATA_DIR) -> Path:
    """Speichert Kalenderdaten partitioniert nach Datum als Parquet-Datei."""
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"calendar_g7_{today}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Gespeichert: %d Events → %s", len(df), path)
    return path


# ── Pipeline ──────────────────────────────────────────────────────────────

def run(
    date_from: str | None = None,
    date_to: str | None = None,
    out_dir: Path = DATA_DIR,
    strict_filter: bool = False,
) -> pd.DataFrame:
    """
    Vollstaendige Pipeline:
    Download → Parsen → G7-Filter → High-Impact-Filter → Parquet
    """
    html = fetch_raw_calendar(date_from=date_from, date_to=date_to)
    if not html:
        logger.error("Kein HTML empfangen — Pipeline abgebrochen")
        return _empty_calendar_df()

    df = parse_calendar_html(html)
    df = filter_g7_currencies(df)
    df = filter_high_impact(df, strict=strict_filter)

    if not df.empty:
        save_to_parquet(df, out_dir)
    else:
        logger.warning("Keine High-Impact Events gefunden — kein Parquet erstellt")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run()
    if result.empty:
        print("Keine Events gefunden.")
    else:
        print("\n=== Wirtschaftskalender — High-Impact Events ===")
        cols = ["date", "time", "currency", "event_name", "impact"]
        print(result[cols].to_string(index=False))
