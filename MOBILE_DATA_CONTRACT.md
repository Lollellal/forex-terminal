# Mobile MVP Data Contract (Implementierungsschritt 7)

Definiert exakt, welche Requests die Mobile App für die 5 MVP-Screens
braucht. Nur Lesezugriffe auf bestehende Projektionen + zwei Schreibpfade,
die es schon gibt (Confirm/Open/Close, JournalNote). Keine neue
Domain-Logik — jeder neue Endpunkt ist reine Aggregation über bereits
vorhandene `projections.*`-Tabellen und `core.accounts.user_id`/
`core.empires.user_id`.

Ziel-Kennzahl: **Home lädt mit genau 1 Request.**

---

## 1. Home (Dashboard)

**Braucht:** Gesamtkapital über alle Accounts (Empire + Standalone), Anzahl
aktiver Trades, letzte Journal-Aktivität — alles auf einen Blick, ohne dass
Mobile vorher wissen muss, welche Empire-/Account-IDs existieren.

**Bestehende API reicht nicht:** Es gab bisher keinen Weg, "alle Accounts/
Empires für user_id X" zu ermitteln — `GET /empires/{id}/overview` und
`GET /accounts/{id}/balance` setzen bereits bekannte IDs voraus. Das ist die
zentrale Lücke, die jeden anderen Screen mit betrifft.

**Neuer Endpunkt:**
```
GET /users/{user_id}/portfolio
```
```json
{
  "user_id": "...",
  "total_balance": "35000.00",
  "total_equity": "34800.00",
  "empires": [
    {"empire_id": "...", "name": "FundingPips", "account_count": 2,
     "total_balance": "25000.00", "total_equity": "24800.00"}
  ],
  "standalone_accounts": [
    {"account_id": "...", "account_type": "LIVE", "status": "ACTIVE",
     "balance": "10000.00", "equity": "10000.00"}
  ],
  "active_trade_count": 3,
  "recent_journal_entries": [
    {"allocation_id": "...", "pair": "EURUSD", "status": "OPEN",
     "updated_at": "2026-07-02T09:14:00Z"}
  ]
}
```
1 Request, keine Vorkenntnis von IDs nötig.

---

## 2. Active Trades

**Braucht:** Alle Allocations mit Status CONFIRMED oder OPEN, über alle
Accounts des Users hinweg (nicht nur einen).

**Bestehende API reicht nicht ganz:** `GET /allocations` filterte bisher nur
nach genau einem `account_id` und genau einem `status` — für "alle aktiven
Trades des Users" hätte Mobile pro Account einen Request gebraucht (N+1)
und hätte CONFIRMED und OPEN getrennt abfragen müssen.

**Erweiterung (kein neuer Endpunkt):**
```
GET /allocations?user_id={id}&status=ACTIVE
```
- `user_id` ist jetzt ein alternativer Filter zu `account_id` (Join über
  `core.accounts.user_id`).
- `status=ACTIVE` ist ein Alias für `status IN ('CONFIRMED','OPEN')`.

1 Request für "alle aktiven Trades des Users".

---

## 3. Journal Update

**Braucht:** Journal-Timeline lesen (über alle Accounts), Notiz zu einer
Allocation hinzufügen/bearbeiten.

**Bestehende API reicht für das Schreiben vollständig:**
`POST /journal-notes` / `PATCH /journal-notes/{id}` ändern nichts — das ist
exakt der in Schritt 5 gebaute Note-Layer, keine neue Regel nötig.

**Erweiterung fürs Lesen (kein neuer Endpunkt):**
```
GET /journal?user_id={id}&status=...
```
Gleiche `user_id`-Join-Erweiterung wie bei `/allocations`, aus demselben
Grund (sonst N+1 über alle Accounts).

---

## 4. Empire Overview

**Braucht:** a) Liste aller Empires des Users mit Kennzahlen (schon in
`/users/{user_id}/portfolio` enthalten — kein Extra-Request nötig, wenn
Mobile von Home reinspringt). b) Beim Antippen einer Empire: welche Accounts
gehören dazu.

**Bestehende API reicht nicht ganz:** `GET /empires/{id}/overview` liefert
nur aggregierte Summen, keine Liste der Mitglieds-Accounts.

**Neuer Endpunkt:**
```
GET /empires/{empire_id}/accounts
```
Liste von `projections.account_balances`-Zeilen mit `empire_id = :id`.

---

## 5. Weekly Report

**Bestehende API reicht nicht — und kann in diesem Schritt nicht ergänzt
werden, ohne die Vorgabe "keine neuen Domain-Regeln" zu verletzen.**

`core.weekly_reports` (BACKEND_ARCHITECTURE.md §2.1) und das komplette
`domain/weekly_report/`-Aggregate (`WeeklyReportGenerated`/`Published`-
Events, Projektion) existieren **noch nicht** — die aktuelle Report-Pipeline
läuft rein auf dem Desktop und legt PDFs lokal unter
`Research weekly reports/` ab, ohne jede Anbindung an `backend/`.

**Vorgeschlagener Ziel-Contract** (nur Spezifikation, nicht implementiert):
```
GET /users/{user_id}/weekly-reports?limit=10
```
```json
[
  {"id": "...", "period_start": "2026-06-22", "period_end": "2026-06-28",
   "status": "PUBLISHED", "content_ref": "https://.../report.pdf",
   "published_at": "2026-06-29T08:00:00Z"}
]
```
**Das ist eine offene Lücke für einen eigenen künftigen Schritt** (WeeklyReport-
Aggregate + Migration zum Hochladen/Referenzieren der PDFs), nicht Teil
dieses API-Layers.

---

## Zusammenfassung: neu vs. bestehend

| Screen | Bestehend ausreichend? | Neu |
|---|---|---|
| Home | Nein | `GET /users/{user_id}/portfolio` |
| Active Trades | Teilweise | `user_id`+`status=ACTIVE`-Filter auf `GET /allocations` |
| Journal Update | Schreiben: ja / Lesen: teilweise | `user_id`-Filter auf `GET /journal` |
| Empire Overview | Teilweise | `GET /empires/{empire_id}/accounts` |
| Weekly Report | Nein, fehlt komplett | Nicht gebaut — offene Lücke, eigener Schritt nötig |

Mit diesen 3 Ergänzungen lädt Home in 1 Request, Active Trades in 1 Request,
Journal in 1 Request, Empire-Detail in 1 Request. Weekly Report bleibt
bewusst offen.
