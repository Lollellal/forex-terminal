"""Tests fuer den Wirtschaftskalender-Fetcher (kein Netzwerk-Zugriff noetig)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data.calendar_fetcher import (
    G7_COUNTRY_IDS,
    HIGH_IMPACT_KEYWORDS,
    _normalize_date,
    _parse_currency,
    _parse_impact,
    filter_g7_currencies,
    filter_high_impact,
    fetch_raw_calendar,
    parse_calendar_html,
    run,
    save_to_parquet,
)
from bs4 import BeautifulSoup


# ── Mock-HTML ─────────────────────────────────────────────────────────────

# Repraesentatives HTML-Fragment wie investing.com es zurueckgibt.
MOCK_HTML = """
<tr class="theDay">Monday, April 28, 2026</tr>

<tr id="eventRowId_100" class="js-event-item" data-country_id="5">
  <td class="first left time js-time">8:30</td>
  <td class="left flagCur noWrap"><span class="ceFlags US"></span> USD</td>
  <td class="left textNum sentiment noWrap">
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
  </td>
  <td class="left event"><a href="#">Nonfarm Payrolls</a></td>
  <td></td>
  <td id="eventActual_100"></td>
  <td id="eventForecast_100">175K</td>
  <td id="eventPrevious_100">228K</td>
</tr>

<tr id="eventRowId_101" class="js-event-item" data-country_id="72">
  <td class="first left time js-time">13:15</td>
  <td class="left flagCur noWrap"><span class="ceFlags EU"></span> EUR</td>
  <td class="left textNum sentiment noWrap">
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
  </td>
  <td class="left event"><a href="#">ECB Interest Rate Decision</a></td>
  <td></td>
  <td id="eventActual_101"></td>
  <td id="eventForecast_101">4.25%</td>
  <td id="eventPrevious_101">4.50%</td>
</tr>

<tr id="eventRowId_102" class="js-event-item" data-country_id="5">
  <td class="first left time js-time">15:00</td>
  <td class="left flagCur noWrap"><span class="ceFlags US"></span> USD</td>
  <td class="left textNum sentiment noWrap">
    <i class="grayFullBullishIcon"></i>
  </td>
  <td class="left event"><a href="#">Building Permits</a></td>
  <td></td>
  <td id="eventActual_102"></td>
  <td id="eventForecast_102">1.45M</td>
  <td id="eventPrevious_102">1.48M</td>
</tr>

<tr class="theDay">Tuesday, April 29, 2026</tr>

<tr id="eventRowId_200" class="js-event-item" data-country_id="4">
  <td class="first left time js-time">9:00</td>
  <td class="left flagCur noWrap"><span class="ceFlags GB"></span> GBP</td>
  <td class="left textNum sentiment noWrap">
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
  </td>
  <td class="left event"><a href="#">CPI y/y</a></td>
  <td></td>
  <td id="eventActual_200"></td>
  <td id="eventForecast_200">3.1%</td>
  <td id="eventPrevious_200">3.4%</td>
</tr>

<tr id="eventRowId_201" class="js-event-item" data-country_id="999">
  <td class="first left time js-time">10:00</td>
  <td class="left flagCur noWrap"><span class="ceFlags XX"></span> XYZ</td>
  <td class="left textNum sentiment noWrap">
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
    <i class="grayFullBullishIcon"></i>
  </td>
  <td class="left event"><a href="#">Non-G7 Event</a></td>
  <td></td>
  <td id="eventActual_201"></td>
  <td id="eventForecast_201"></td>
  <td id="eventPrevious_201"></td>
</tr>
"""

# HTML mit expliziter impact-N Klasse auf der Zeile (alternative Struktur)
MOCK_HTML_IMPACT_CLASS = """
<tr class="theDay">Monday, April 28, 2026</tr>
<tr id="eventRowId_300" class="js-event-item impact-3" data-country_id="5">
  <td class="first left time js-time">14:00</td>
  <td class="left flagCur noWrap"> USD</td>
  <td class="left textNum sentiment noWrap"></td>
  <td class="left event"><a href="#">FOMC Statement</a></td>
  <td></td>
  <td id="eventActual_300"></td>
  <td id="eventForecast_300"></td>
  <td id="eventPrevious_300"></td>
</tr>
<tr id="eventRowId_301" class="js-event-item impact-1" data-country_id="5">
  <td class="first left time js-time">16:00</td>
  <td class="left flagCur noWrap"> USD</td>
  <td class="left textNum sentiment noWrap"></td>
  <td class="left event"><a href="#">Durable Goods Orders m/m</a></td>
  <td></td>
  <td id="eventActual_301"></td>
  <td id="eventForecast_301"></td>
  <td id="eventPrevious_301"></td>
</tr>
"""


# ── parse_calendar_html ────────────────────────────────────────────────────

def test_parse_liefert_alle_felder():
    df = parse_calendar_html(MOCK_HTML)
    expected_cols = {"date", "time", "currency", "event_name", "impact", "forecast", "previous"}
    assert expected_cols.issubset(set(df.columns))


def test_parse_korrekte_zeilenanzahl():
    df = parse_calendar_html(MOCK_HTML)
    # 5 Event-Zeilen insgesamt (inkl. Non-G7-Event)
    assert len(df) == 5


def test_parse_datum_wechselt_korrekt():
    df = parse_calendar_html(MOCK_HTML)
    # Events 0-2 gehoeren zu Montag, Events 3-4 zu Dienstag
    assert df.loc[0, "date"] == "2026-04-28"
    assert df.loc[1, "date"] == "2026-04-28"
    assert df.loc[2, "date"] == "2026-04-28"
    assert df.loc[3, "date"] == "2026-04-29"
    assert df.loc[4, "date"] == "2026-04-29"


def test_parse_event_name_korrekt():
    df = parse_calendar_html(MOCK_HTML)
    assert df.loc[0, "event_name"] == "Nonfarm Payrolls"
    assert df.loc[1, "event_name"] == "ECB Interest Rate Decision"


def test_parse_forecast_und_previous():
    df = parse_calendar_html(MOCK_HTML)
    assert df.loc[0, "forecast"] == "175K"
    assert df.loc[0, "previous"] == "228K"
    assert df.loc[1, "forecast"] == "4.25%"


def test_parse_time_korrekt():
    df = parse_calendar_html(MOCK_HTML)
    assert df.loc[0, "time"] == "8:30"
    assert df.loc[1, "time"] == "13:15"


def test_parse_leeres_html_gibt_leeren_df():
    df = parse_calendar_html("")
    assert df.empty
    assert "event_name" in df.columns


def test_parse_whitespace_html_gibt_leeren_df():
    df = parse_calendar_html("   \n  ")
    assert df.empty


def test_parse_keine_event_zeilen_gibt_leeren_df():
    html = "<tr class='theDay'>Monday, April 28, 2026</tr>"
    df = parse_calendar_html(html)
    assert df.empty


# ── _parse_impact ──────────────────────────────────────────────────────────

def _row_with_icons(n: int, extra_classes: str = "") -> BeautifulSoup:
    icons = "".join('<i class="grayFullBullishIcon"></i>' for _ in range(n))
    html = f"""
    <tr class="js-event-item {extra_classes}">
      <td class="sentiment">{icons}</td>
    </tr>"""
    return BeautifulSoup(html, "lxml").find("tr")


def test_impact_drei_icons_ist_high():
    row = _row_with_icons(3)
    assert _parse_impact(row) == "high"


def test_impact_zwei_icons_ist_medium():
    row = _row_with_icons(2)
    assert _parse_impact(row) == "medium"


def test_impact_ein_icon_ist_low():
    row = _row_with_icons(1)
    assert _parse_impact(row) == "low"


def test_impact_klasse_impact_3_ist_high():
    row = _row_with_icons(0, extra_classes="impact-3")
    assert _parse_impact(row) == "high"


def test_impact_klasse_impact_1_ist_low():
    row = _row_with_icons(0, extra_classes="impact-1")
    assert _parse_impact(row) == "low"


def test_impact_kein_hinweis_ist_unknown():
    html = '<tr class="js-event-item"><td class="sentiment"></td></tr>'
    row = BeautifulSoup(html, "lxml").find("tr")
    assert _parse_impact(row) == "unknown"


def test_parse_impact_class_auf_zeile():
    df = parse_calendar_html(MOCK_HTML_IMPACT_CLASS)
    fomc_row = df[df["event_name"] == "FOMC Statement"].iloc[0]
    assert fomc_row["impact"] == "high"

    dgo_row = df[df["event_name"] == "Durable Goods Orders m/m"].iloc[0]
    assert dgo_row["impact"] == "low"


# ── _parse_currency ────────────────────────────────────────────────────────

def _row_with_currency(text: str, country_id: str | None = None) -> BeautifulSoup:
    cid_attr = f'data-country_id="{country_id}"' if country_id else ""
    html = f"""
    <tr class="js-event-item" {cid_attr}>
      <td class="left flagCur noWrap">{text}</td>
    </tr>"""
    return BeautifulSoup(html, "lxml").find("tr")


def test_parse_currency_aus_text():
    row = _row_with_currency("USD")
    assert _parse_currency(row) == "USD"


def test_parse_currency_mit_span():
    row = _row_with_currency('<span class="ceFlags US"></span> USD')
    assert _parse_currency(row) == "USD"


def test_parse_currency_fallback_country_id():
    # Kein Currency-Text, aber gueltiger country_id
    html = '<tr class="js-event-item" data-country_id="5"><td class="flagCur"></td></tr>'
    row = BeautifulSoup(html, "lxml").find("tr")
    assert _parse_currency(row) == "USD"


def test_parse_currency_unbekannt():
    html = '<tr class="js-event-item"><td class="flagCur">xyz</td></tr>'
    row = BeautifulSoup(html, "lxml").find("tr")
    # 'xyz' ist 3 Kleinbuchstaben – wird nicht als Waehrungscode erkannt
    # (regex \b[A-Z]{3}\b matcht nur Grossbuchstaben)
    assert _parse_currency(row) == "UNKNOWN"


# ── _normalize_date ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("Monday, April 28, 2026", "2026-04-28"),
    ("April 28, 2026",          "2026-04-28"),
    ("2026-04-28",              "2026-04-28"),
    ("28/04/2026",              "2026-04-28"),
])
def test_normalize_date_formate(raw, expected):
    assert _normalize_date(raw) == expected


def test_normalize_date_unbekanntes_format_gibt_original():
    result = _normalize_date("kein-datum")
    assert result == "kein-datum"


# ── filter_g7_currencies ───────────────────────────────────────────────────

def test_filter_g7_entfernt_non_g7():
    df = parse_calendar_html(MOCK_HTML)
    result = filter_g7_currencies(df)
    assert "XYZ" not in result["currency"].values
    assert "UNKNOWN" not in result["currency"].values


def test_filter_g7_behaelt_alle_g7():
    df = parse_calendar_html(MOCK_HTML)
    result = filter_g7_currencies(df)
    for currency in result["currency"]:
        assert currency in G7_COUNTRY_IDS


def test_filter_g7_leerer_df_bleibt_leer():
    from src.data.calendar_fetcher import _empty_calendar_df
    df = _empty_calendar_df()
    result = filter_g7_currencies(df)
    assert result.empty


# ── filter_high_impact ────────────────────────────────────────────────────

def test_filter_high_impact_strict_mode():
    df = parse_calendar_html(MOCK_HTML)
    g7_df = filter_g7_currencies(df)
    result = filter_high_impact(g7_df, strict=True)
    assert all(result["impact"] == "high")


def test_filter_high_impact_keyword_fallback():
    """Events mit impact='unknown' aber High-Impact-Stichwort werden behalten."""
    df = pd.DataFrame([
        {"date": "2026-04-28", "time": "8:30", "currency": "USD",
         "event_name": "Nonfarm Payrolls", "impact": "unknown",
         "forecast": "", "previous": ""},
        {"date": "2026-04-28", "time": "10:00", "currency": "USD",
         "event_name": "Building Permits", "impact": "unknown",
         "forecast": "", "previous": ""},
    ])
    result = filter_high_impact(df, strict=False)
    # "Nonfarm Payrolls" enthaelt "nonfarm payroll" → keyword match
    assert any("Nonfarm" in name for name in result["event_name"])
    # "Building Permits" ist kein High-Impact-Keyword → wird entfernt
    assert not any("Building" in name for name in result["event_name"])


def test_filter_high_impact_entfernt_low_ohne_keyword():
    df = pd.DataFrame([{
        "date": "2026-04-28", "time": "15:00", "currency": "USD",
        "event_name": "Building Permits", "impact": "low",
        "forecast": "", "previous": "",
    }])
    result = filter_high_impact(df, strict=False)
    assert result.empty


def test_filter_high_impact_leerer_df_bleibt_leer():
    from src.data.calendar_fetcher import _empty_calendar_df
    df = _empty_calendar_df()
    assert filter_high_impact(df).empty


# ── save_to_parquet ────────────────────────────────────────────────────────

def test_save_to_parquet_erstellt_datei():
    df = parse_calendar_html(MOCK_HTML)
    df = filter_g7_currencies(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_to_parquet(df, Path(tmpdir))
        assert path.exists()
        assert path.suffix == ".parquet"


def test_save_to_parquet_dateiname_enthaelt_prefix():
    df = parse_calendar_html(MOCK_HTML)
    df = filter_g7_currencies(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_to_parquet(df, Path(tmpdir))
        assert "calendar_g7_" in path.name


def test_save_to_parquet_daten_runden_trip():
    df = parse_calendar_html(MOCK_HTML)
    df = filter_g7_currencies(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_to_parquet(df, Path(tmpdir))
        loaded = pd.read_parquet(path)
        assert len(loaded) == len(df)
        assert set(loaded.columns) == set(df.columns)


# ── fetch_raw_calendar (mit Mock) ─────────────────────────────────────────

def _mock_response(html_fragment: str, status_code: int = 200) -> MagicMock:
    """Erstellt ein Mock-Response-Objekt fuer requests.post."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = {"data": html_fragment, "rows_num": 5}
    mock.raise_for_status.return_value = None
    return mock


@patch("src.data.calendar_fetcher.requests.post")
def test_fetch_gibt_html_fragment_zurueck(mock_post):
    mock_post.return_value = _mock_response(MOCK_HTML)
    result = fetch_raw_calendar(date_from="2026-04-28", date_to="2026-05-05")
    assert result == MOCK_HTML
    assert mock_post.called


@patch("src.data.calendar_fetcher.time.sleep")
@patch("src.data.calendar_fetcher.requests.post")
def test_fetch_retry_bei_timeout(mock_post, mock_sleep):
    """Nach Timeout wird ein weiterer Versuch unternommen."""
    mock_post.side_effect = [
        requests.exceptions.Timeout("Timeout"),
        _mock_response(MOCK_HTML),
    ]
    result = fetch_raw_calendar(date_from="2026-04-28", date_to="2026-05-05", max_retries=3)
    assert result == MOCK_HTML
    assert mock_post.call_count == 2


@patch("src.data.calendar_fetcher.time.sleep")
@patch("src.data.calendar_fetcher.requests.post")
def test_fetch_gibt_leer_nach_allen_fehlern(mock_post, mock_sleep):
    """Alle Retries fehlgeschlagen → leerer String."""
    import requests as req_module
    mock_post.side_effect = req_module.exceptions.ConnectionError("offline")
    result = fetch_raw_calendar(date_from="2026-04-28", date_to="2026-05-05", max_retries=2)
    assert result == ""
    assert mock_post.call_count == 2


@patch("src.data.calendar_fetcher.requests.post")
def test_fetch_gibt_leer_bei_leerer_antwort(mock_post):
    """API antwortet mit leerem 'data'-Feld → leerer String."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": "", "rows_num": 0}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    result = fetch_raw_calendar(date_from="2026-04-28", date_to="2026-05-05", max_retries=1)
    assert result == ""


# ── run (Integrations-Pipeline) ────────────────────────────────────────────

@patch("src.data.calendar_fetcher.requests.post")
def test_run_pipeline_end_to_end(mock_post):
    mock_post.return_value = _mock_response(MOCK_HTML)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(
            date_from="2026-04-28",
            date_to="2026-05-05",
            out_dir=Path(tmpdir),
        )

    assert not result.empty
    assert set(result.columns) >= {"date", "time", "currency", "event_name", "impact"}
    # Alle zurueckgegebenen Events muessen G7-Waehrungen sein
    assert all(c in G7_COUNTRY_IDS for c in result["currency"])


@patch("src.data.calendar_fetcher.requests.post")
def test_run_erstellt_parquet(mock_post):
    mock_post.return_value = _mock_response(MOCK_HTML)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir)
        run(date_from="2026-04-28", date_to="2026-05-05", out_dir=out_path)
        parquet_files = list(out_path.glob("*.parquet"))
        assert len(parquet_files) == 1


@patch("src.data.calendar_fetcher.time.sleep")
@patch("src.data.calendar_fetcher.requests.post")
def test_run_bei_netzwerkfehler_gibt_leeren_df(mock_post, mock_sleep):
    import requests as req_module
    mock_post.side_effect = req_module.exceptions.ConnectionError("offline")

    with tempfile.TemporaryDirectory() as tmpdir:
        result = run(
            date_from="2026-04-28",
            date_to="2026-05-05",
            out_dir=Path(tmpdir),
            strict_filter=True,
        )

    assert result.empty


# ── HIGH_IMPACT_KEYWORDS Vollstaendigkeit ──────────────────────────────────

@pytest.mark.parametrize("event", [
    "Nonfarm Payrolls",
    "Consumer Price Index",
    "FOMC Statement",
    "ECB Press Conference",
    "GDP q/q",
    "Unemployment Rate",
    "Retail Sales m/m",
    "ISM Manufacturing PMI",
    "Average Hourly Earnings m/m",
    "Initial Jobless Claims",
    "Core PCE Price Index",
])
def test_keyword_trifft_standard_events(event):
    lower = event.lower()
    matched = any(kw in lower for kw in HIGH_IMPACT_KEYWORDS)
    assert matched, f"'{event}' sollte durch Keyword-Liste getroffen werden"
