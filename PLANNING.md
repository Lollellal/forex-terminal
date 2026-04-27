# FOREX ML TERMINAL — Project Planning Document
Version 1.0 | April 2026

---

## 1. Zweck des Terminals

Das Terminal unterstuetzt den Trader bei der Einschaetzung des makrooekonomischen Bias pro Waehrung.
Es ersetzt keine eigene Analyse, sondern liefert eine datengetriebene Zweitmeinung auf Basis von CoT-Daten und Makrodaten.

**Der Trader bringt selbst mit:**
- Technische Analyse
- Saisonalitaeten
- Finalen Trade-Entscheid

**Das Terminal liefert:**
- CoT-Daten (CFTC) aufbereitet und gewichtet
- Makrodaten (Zinsen, CPI, PMI, Arbeitsmarkt, BIP etc.)
- Wirtschaftskalender mit High-Impact Events
- Gewichteten Gesamt-Bias pro Waehrung mit Erklaerung warum

---

## 2. Abgedeckte Waehrungen

G7 Majors: USD | EUR | GBP | JPY | CAD | CHF | AUD

---

## 3. System-Architektur

### 3.1 Tech Stack
- **Frontend:** Next.js 14 (App Router) + TypeScript + Tailwind CSS
- **Backend:** Python FastAPI
- **Datenbank:** PostgreSQL (historische Daten) + Redis (Cache)
- **ML:** scikit-learn + pandas
- **Scheduler:** APScheduler (automatische Datenupdates)

### 3.2 Datenquellen

| Quelle | Daten | Update-Frequenz | Kosten |
|--------|-------|-----------------|--------|
| CFTC.gov | CoT Positioning | Woechentlich (Freitag) | Kostenlos |
| FRED (St. Louis Fed) | Zinsen, CPI, BIP, PMI | Monatlich/Quartalsweise | Kostenlos |
| investing.com / ForexFactory | Wirtschaftskalender | Taeglich | Kostenlos |
| Yahoo Finance / Alpha Vantage | OHLCV Forex Daten | Taeglich | Kostenlos |
| FRED / Bloomberg Proxy | Anleihe Yields, VIX | Taeglich | Kostenlos |

---

## 4. ML Modelle

Jedes Modell bewertet eine spezifische Dimension und gibt als Output:
**Richtung** (Bullish / Bearish / Neutral) + **Konfidenz** (0–100%)

---

### Gruppe A: Zinspolitik & Zentralbank (Universalgewicht: 25%)

| # | Modell | Input-Features | Waehrungsmodifikator | Konfidenz-Typ |
|---|--------|---------------|----------------------|---------------|
| A1 | Leitzins Absolut | Aktueller Leitzins, historischer Durchschnitt, Ziel-Rate | Alle gleich | Hoch |
| A2 | Zinsdifferenz | Leitzins-Spread vs. USD | Alle gleich | Hoch |
| A3 | Zinserwartung | Rate Futures, OIS Swap: wie viele Cuts/Hikes erwartet? | USD x1.5 | Sehr Hoch |
| A4 | ZB Haltung (Hawk/Dove) | Score aus letzter Pressekonferenz | Alle gleich | Mittel |
| A5 | Zinsueberraschung | Erwarteter vs. tatsaechlicher Zinsentscheid | Alle gleich | Sehr Hoch (Event) |

---

### Gruppe B: Inflation (Universalgewicht: 12%)

| # | Modell | Input-Features | Waehrungsmodifikator | Konfidenz-Typ |
|---|--------|---------------|----------------------|---------------|
| B1 | CPI Absolut vs. Ziel | CPI YoY, Core CPI, ZB-Ziel (meist 2%) | Alle gleich | Hoch |
| B2 | CPI Trend | CPI Veraenderung letzte 3/6 Monate | Alle gleich | Mittel |
| B3 | CPI Surprise | Erwarteter vs. tatsaechlicher CPI | Alle gleich | Sehr Hoch (Event) |
| B4 | CPI Differenz | CPI dieser Waehrung vs. USD CPI relativ | Alle gleich | Hoch |

---

### Gruppe C: Arbeitsmarkt (Universalgewicht: 6%)

| # | Modell | Input-Features | Waehrungsmodifikator | Konfidenz-Typ |
|---|--------|---------------|----------------------|---------------|
| C1 | Arbeitslosenquote | Aktuelle Rate, Trend, historischer Durchschnitt | USD x1.5 | Mittel |
| C2 | Beschaeftigung Surprise | Erwartete vs. tatsaechliche Jobzahlen (NFP, ADP) | USD x2.0 | Sehr Hoch (Event) |
| C3 | Lohnwachstum | Average Hourly Earnings YoY | USD x1.5 | Hoch |

---

### Gruppe D: Wachstum & PMI (Universalgewicht: 10%)

| # | Modell | Input-Features | Waehrungsmodifikator | Konfidenz-Typ |
|---|--------|---------------|----------------------|---------------|
| D1 | BIP Wachstum | BIP QoQ, YoY, Trend, Revision | Alle gleich | Mittel |
| D2 | PMI Manufacturing | PMI Wert (50 = Grenze), Trend, Surprise | Alle gleich | Hoch |
| D3 | PMI Services | Services PMI, Composite PMI | EUR x1.3 | Hoch |
| D4 | Retail Sales | Konsumausgaben Trend, YoY Veraenderung | USD x1.3 | Mittel |
| D5 | Wachstum Relativ | BIP-Wachstum dieser Waehrung vs. USD | Alle gleich | Hoch |

---

### Gruppe E: CoT & Positionierung (Universalgewicht: 20%)

| # | Modell | Input-Features | Waehrungsmodifikator | Konfidenz-Typ |
|---|--------|---------------|----------------------|---------------|
| E1 | CoT Net Position | Non-Commercial Net Longs, Trend letzte 4 Wochen | Alle gleich | Hoch |
| E2 | CoT Perzentil | Historisches Perzentil der Net Position (3-Jahres-Fenster) | Alle gleich | Sehr Hoch |
| E3 | CoT Wochen-Flow | Wochenveraenderung: beschleunigt oder verlangsamt? | Alle gleich | Hoch |
| E4 | CoT Extremwert-Kontra | Bei >90% oder <10% Perzentil: Kontraindikator | Alle gleich | Sehr Hoch |

---

### Gruppe F: Kapitalfluesse & Makro-Surprise (Universalgewicht: 15%)

| # | Modell | Input-Features | Waehrungsmodifikator | Konfidenz-Typ |
|---|--------|---------------|----------------------|---------------|
| F1 | Carry Trade Attraktivitaet | Zinsdifferenz + Risk-Umfeld | JPY x2.0 (Carry-Funding) | Hoch |
| F2 | Economic Surprise Index | Citi ESI: kommen Daten besser oder schlechter als erwartet? | Alle gleich | Sehr Hoch |
| F3 | Handelsbilanz / Current Account | Leistungsbilanzsaldo, Trend | JPY x1.5, CHF x1.3 | Mittel |
| F4 | Anleihe-Yield Spread | 10Y Yield dieser Waehrung vs. US 10Y | USD x2.0 | Hoch |

---

### Gruppe G: Risikoumfeld & Safe Haven (Universalgewicht: 8%)

| # | Modell | Input-Features | Waehrungsmodifikator | Konfidenz-Typ |
|---|--------|---------------|----------------------|---------------|
| G1 | VIX / Risk-off | VIX Level und Trend: hoch = Risk-off | JPY x3.0, CHF x3.5, USD x1.5 | Sehr Hoch |
| G2 | Equity Trend | S&P500 Trend: Risk-on oder Risk-off Regime? | AUD x1.5, JPY x2.0 | Hoch |
| G3 | Gold Trend | Gold Preis-Trend als Risk-off Indikator | CHF x1.5, JPY x1.3 | Mittel |
| G4 | Yield Curve (2Y-10Y) | Invertiert = Rezessionsrisiko = Risk-off Druck | USD x2.0 | Hoch |

---

### Gruppe H: Rohstoffe (Universalgewicht: 4%)

| # | Modell | Input-Features | Waehrungsmodifikator | Konfidenz-Typ |
|---|--------|---------------|----------------------|---------------|
| H1 | Oel (WTI/Brent) | Oel-Preis Trend, Wochenveraenderung, 50/200 MA | CAD x3.5 (dominiert!) | Sehr Hoch fuer CAD |
| H2 | Kupfer & Industriemetalle | Kupfer-Trend als China/Wachstums-Proxy | AUD x2.5, NZD x1.5 | Hoch |
| H3 | Gold | Gold-Trend (auch AUD-Korrelation) | AUD x2.0 | Mittel fuer AUD |

---

## 5. Gewichtungs-System

### 5.1 Universalgewichte

| Rang | Gruppe | Universalgewicht | Begruendung |
|------|--------|-----------------|-------------|
| 1 | A: Zinspolitik & Zentralbank | 25% | Haupttreiber von Kapitalfluessen |
| 2 | E: CoT & Positionierung | 20% | Zeigt wo Smart Money steht |
| 3 | F: Kapitalfluesse & Surprise | 15% | Relative Staerke und Ueberraschungen |
| 4 | B: Inflation | 12% | Zwingt ZB zu handeln |
| 5 | D: Wachstum & PMI | 10% | Vorlaufindikator Wirtschaftsmomentum |
| 6 | G: Risikoumfeld | 8% | Safe Haven Flows |
| 7 | C: Arbeitsmarkt | 6% | Wichtig aber oft in CPI/PMI gespiegelt |
| 8 | H: Rohstoffe | 4% | Relevant nur fuer CAD/AUD |
| | **TOTAL** | **100%** | |

### 5.2 Score-Formel

```
Score = Sum( Richtung x Konfidenz x Universalgewicht x Waehrungsmodifikator )
```

- **Richtung:** +1 (Bullish), 0 (Neutral), -1 (Bearish)
- **Konfidenz:** 0.0 bis 1.0 (historische Backtest-Genauigkeit des Modells)

**Score-Interpretation:**
- `> +0.3` → BULLISH (stark)
- `+0.1 bis +0.3` → BULLISH (schwach)
- `-0.1 bis +0.1` → NEUTRAL
- `-0.1 bis -0.3` → BEARISH (schwach)
- `< -0.3` → BEARISH (stark)

### 5.3 Dynamische Regime-Gewichtung

| Regime | Erkennung | Hochgewichtet | Runtergewichtet |
|--------|-----------|---------------|-----------------|
| Risk-off | VIX > 25, Equity faellt | G: Risikoumfeld x2.0 | H: Rohstoffe x0.3 |
| Risk-on | VIX < 15, Equity steigt | D: Wachstum x1.5 | G: Risikoumfeld x0.5 |
| Zinswendephase | ZB wechselt Kurs | A: Zinspolitik x2.0 | H: Rohstoffe x0.5 |
| Rohstoff-Rally | Oel/Kupfer +15% in 4W | H: Rohstoffe x3.0 | B: Inflation x0.7 |

---

## 6. Backtesting-Methode

### 6.1 Walk-Forward Validation
- Training: 2 Jahre historische Daten
- Validation: 6 Monate (Hyperparameter-Tuning)
- Test (Out-of-Sample): 6 Monate
- Fenster rollt jeweils 3 Monate vor

### 6.2 Metriken

| Metrik | Zielwert | Bedeutung | Warnschwelle |
|--------|----------|-----------|--------------|
| Directional Accuracy | > 55% | Richtige Richtung in % der Faelle | < 50% = schlechter als Zufall |
| Konfidenz-Kalibrierung | Hoch = richtiger | 80% Konfidenz sollte 80% stimmen | Grosse Abweichung = neu trainieren |
| Sharpe Ratio | > 1.0 | Risk-adjusted Return | < 0.5 = ueberpruefen |
| Max Drawdown | < 20% | Schlechteste Verlustphase | > 30% = Modell-Problem |

---

## 7. Terminal UI Layout

### 7.1 Aufbau (wie Delta Advisory)
- **Linke Seite:** Scatter Plot (X-Achse = CoT-Staerke, Y-Achse = Makro-Score)
- **Rechte Seite:** Waehrungs-Tabelle mit Bias, Modellanzahl, Long-Ratio
- **Klick auf Waehrung:** Detail-Ansicht mit allen Modell-Scores + Erklaerung
- **Oben:** Wirtschaftskalender-Banner mit High-Impact Events dieser Woche
- **Farbschema:** Dunkel (Terminal-Style wie Delta Advisory)

### 7.2 Bias-Anzeige pro Waehrung

```
EUR  | BULLISH  | Bias aus 6 Modellen  |  Bullish - 75% Long
     | Haupt-Treiber: Zinsdifferenz (stark), CoT-Flow (mittel)
     | Gegenargumente: CPI faellt (schwach bearish)
     | Warnung: Score und Modellmehrheit widersprechen sich!
     | Event: EZB Entscheidung Donnerstag
```

---

## 8. Baureihenfolge fuer Claude Code

> **WICHTIG:** Immer diese Datei zuerst einlesen lassen:
> `"Lies zuerst PLANNING.md, dann machen wir Schritt X"`

| # | Schritt | Aufgabe | Output |
|---|---------|---------|--------|
| 1 | Datenpipeline CoT | CFTC-Daten fetchen, parsen, speichern | /data/raw/cot/ |
| 2 | Datenpipeline Makro | FRED API: Zinsen, CPI, PMI, BIP fetchen | /data/raw/macro/ |
| 3 | Datenpipeline Kalender | Wirtschaftskalender scrapen/API | /data/raw/calendar/ |
| 4 | Feature Engineering | Alle Modell-Features berechnen | /data/features/ |
| 5 | ML Modelle (Gruppe A) | Zinsmodelle bauen + backtesten | /models/group_a/ |
| 6 | ML Modelle (B-H) | Restliche Modellgruppen | /models/ |
| 7 | Regime-Erkennung | Marktregime identifizieren + Gewichte anpassen | /models/regime/ |
| 8 | Score-Aggregation | Gewichteten Gesamt-Score berechnen | /scoring/ |
| 9 | FastAPI Backend | REST Endpoints fuer alle Daten | /backend/ |
| 10 | Next.js Frontend | Terminal UI bauen | /frontend/ |

---

## 9. Wichtige Regeln fuer Claude Code

- Immer Python 3.11 verwenden
- Daten als Parquet speichern (nicht CSV) fuer Performance
- Nach jedem Schritt einen einfachen Test schreiben
- Keine Hardcoded API Keys — immer .env Datei verwenden
- Jede Funktion kurz erklaeren bevor weitermachen
- Bei Fehler: vollstaendige Fehlermeldung einfuegen, nicht paraphrasieren
- Nie das ganze Projekt auf einmal — immer ein Schritt nach dem anderen
