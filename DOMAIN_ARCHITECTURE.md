# Trading Operating System — Domain Architecture (Phase 1)

_Status: **EINGEFROREN** (Runde 3) — Grundlage für Phase 2 (Schema/Backend-Ableitung)_
_Scope: Nur fachliche Architektur. Kein Code, kein Schema, keine API._

---

## 0. Leitplanken

1. Eine Single Source of Truth — keine Client-eigenen Daten.
2. Desktop und Mobile sind gleichberechtigte, logiklose Clients.
3. Genau eine zentrale Domäne — keine Duplikat-Business-Logik.
4. Domain-Driven: Fachlogik vor Datenbank, vor UI.
5. Event-Driven: Domänen kommunizieren nur über Events, nie durch direkte Aufrufe.
6. Command/Event-Trennung: Commands sind Absichten, Events sind Fakten.
7. **Drei-Ebenen-Trennung ist explizit, nicht implizit** (neu, Runde 2):

```
1. COMMAND LAYER (Intent)       — User/System will etwas tun. Desktop + Mobile senden gleichberechtigt.
2. DOMAIN LAYER (Truth)          — Aggregates validieren, entscheiden, mutieren exklusiv ihren eigenen Zustand.
3. PROJECTION LAYER (Views)      — Journal, Portfolio, PerformanceSnapshot, Empire-Übersicht. Rein lesend, aus Events abgeleitet.
```

Diese Trennung ist die wichtigste Korrektur aus Runde 2. Jede Entity aus Runde 1 wird jetzt einer dieser drei Ebenen fest zugeordnet — Vermischung ist nicht erlaubt.

**Bounded Contexts (korrigiert):**

| Context | Rolle | Ebene |
|---|---|---|
| **Research** | Signal-/Playbook-Erzeugung | Domain (Aggregates) |
| **Execution** | TradeAllocation-Lebenszyklus | Domain (Aggregates) |
| **Risk** | Gate (sync, vor Execution) + Monitor (async, auf laufende Events) | Domain — **kein Observer-Context, sondern Pflichtschicht** |
| **Account & Empire** | Kapitalstruktur, Prop-Firm-Lifecycle | Domain (Aggregates) |
| **Calendar & Market** | Externe Fakten | Domain (Aggregate, aber ohne Invarianten ggü. anderen Contexts) |
| **Journal** | Menschlich lesbare Historie | **Projection**, nicht Domain (korrigiert) |
| **Portfolio & Performance** | Aggregierte KPI-/Kapitalsicht | **Projection**, nicht Domain (korrigiert) |

Risk ist bewusst nicht mehr als "Context neben Execution" gelistet, sondern als **Pflichtschicht davor**: Execution darf technisch nicht existieren, ohne dass ein Command zuerst den Risk Gate passiert hat. Details in Abschnitt 3.

---

## 1. Aggregates (Domain Layer — exklusive Schreibautorität)

Ein Aggregate ist die einzige Instanz, die ihren eigenen Zustand mutieren und Invarianten durchsetzen darf. Alles, was hier nicht steht (Journal, Portfolio, PerformanceSnapshot), ist Projection (Abschnitt 1b).

### 1.1 Signal (Research)
Unverändert zu Runde 1: immutable nach Erzeugung, GENERATED → EXPIRED. Erzeugt `SignalGenerated`. Reagiert auf nichts — entsteht aus `GenerateSignalsCommand`.

### 1.2 Playbook (Research)
Unverändert: DRAFT → ACTIVE → RETIRED. Playbook-Vorschläge aus `WeeklyReportPublished` bleiben **Vorschlag, nie Automatik** (bestätigt in Runde 2, Punkt zu Report-Qualität).

### 1.3 TradeAllocation (Execution)
Unverändert im Lebenszyklus (PROPOSED → CONFIRMED → OPEN → PARTIAL_HIT → BREAK_EVEN → CLOSED / CANCELLED).

**Neu (Runde 2, präzisiert Runde 3):** Jeder Zustandsübergang, der Kapital exponiert (`PROPOSED → CONFIRMED`, `CONFIRMED → OPEN`), **muss** synchron durch den Risk Gate laufen, bevor das Aggregate den Command ausführt — und zwar **zweimal unabhängig**: einmal bei `ConfirmAllocationCommand` (geplantes Risiko) und erneut bei `MarkAllocationOpenedCommand` (Ist-Zustand bei tatsächlichem Fill, da Preis/Spread/Account-Status zwischen Plan und Ausführung abweichen können). Kein `AllocationOpened`-Event ohne aktuell gültige `RiskGateDecision: ALLOW` aus Durchlauf 2.

### 1.4 Account (Account & Empire)
Unverändert. Einzige Instanz, die Balance/Equity mutieren darf — auch der Risk Monitor mutiert Account nie direkt, sondern schickt einen Command.

### 1.5 Empire (Account & Empire)
Aggregiert über Accounts, besitzt keine eigene Balance-Wahrheit — liest nur.

**Bestätigt (Runde 3) — keine Automatik bei Stage-Progression:** `EmpireStageAdvanced` entsteht **niemals** automatisch aus einem Risk-Monitor-Trigger (z.B. Challenge Target erreicht). Der Monitor erzeugt nur einen Vorschlag (`EmpireStageAdvanceProposed`), sichtbar z.B. als "Phase 1 vermutlich bestanden – bestätigen?". Erst eine explizite manuelle `ConfirmEmpireStageAdvanceCommand`-Aktion des Users führt zu `EmpireStageAdvanced`. Grund: Prop-Firm-Regeln, Payouts und Dashboards sind zu folgenreich für eine blinde automatische Zustandsänderung — dieselbe Kuration-statt-Automatik-Regel wie bei Playbook-Vorschlägen und Reports gilt hier ebenso.

### 1.6 RiskPolicy (Risk)
**Erweitert (Runde 2):** Neues Pflichtattribut `EvaluationMode: SYNC_GATE | ASYNC_MONITOR` (Details Abschnitt 3).

### 1.7 CalendarEvent (Calendar & Market)
Unverändert. Reine Faktenquelle, keine Invarianten ggü. anderen Aggregates.

### 1.8 WeeklyReport (Research)
Unverändert. DRAFT → GENERATED → PUBLISHED → ARCHIVED, entsteht nur über expliziten `GenerateWeeklyReportCommand`.

### 1.9 JournalNote (neu, Runde 2 — ersetzt "JournalEntry als Entity")

**Verantwortung:** Minimales Schreib-Aggregate für exakt einen Zweck — einen frei formulierten menschlichen Kommentar an eine TradeAllocation oder ein Signal zu hängen. **Kein** Journal-Business-Logic, keine Aggregation, keine KPIs. Löst den Widerspruch "Journal ist keine Entity" vs. "der User tippt aber tatsächlich Text ein": Der Schreibvorgang ist real, aber trivial und ohne Invarianten — daher ein Mikro-Aggregate, kein vollwertiger Context.

**Bestätigt (Runde 3) — harte Grenze für JournalNote:** Ausschließlich subjektiver Kommentar-Layer — Freitext, Screenshots, Emotionen, Lessons Learned. **Niemals** Trade-Fakten (Ergebnis, R-Multiple, Status, Preise). Alles, was faktisch/hart ist, kommt ausschließlich aus Events der jeweiligen Aggregates (TradeAllocation, Account) und wird in der Journal-Projection dazugemischt — nie in JournalNote selbst gespeichert. Damit bleibt JournalNote garantiert konfliktfrei mit der harten Wahrheit: Es kann nie mit Account/Allocation-Daten divergieren, weil es diese Daten gar nicht führt.

**Lebenszyklus:** `CREATED → (EDITED)*`. Kein komplexer State.

**Beziehungen:** Gehört zu genau einer TradeAllocation ODER einem Signal (nie beides).

**Events:** `JournalNoteAdded`, `JournalNoteEdited` — beide mit `aggregate_type: JournalNote`, aber tragen `related_allocation_id` bzw. `related_signal_id` im Payload, damit die Journal-Projection sie korrekt einsortieren kann.

Die **Journal-Projection** (1b) ist der einzige Ort, an dem `JournalNoteAdded` zusammen mit `AllocationClosed`, `AllocationOpened` etc. zu der bekannten Zwei-Spur-Ansicht (Live-Journal / Signal-Journal) zusammengeführt wird. Damit bleibt "Journal" im Sinne des Users vollständig erhalten — nur die Schreibautorität liegt korrekt bei den Aggregates, nicht bei einer Journal-Entity.

---

## 1b. Projections (Query Layer — rein lesend, kein eigener Write-Pfad)

### Journal
Materialisierte Sicht aus `JournalNoteAdded/Edited`, `AllocationOpened`, `AllocationPartialHit`, `AllocationBreakEvenActivated`, `AllocationClosed`, `SignalGenerated`. Erzeugt selbst **keine** Events — sie ist Konsument, nicht Quelle. Zwei Sichten (Live/Signal-Journal) sind zwei **Filter/Views auf denselben Event-Strom**, keine getrennten Datenmodelle.

### Portfolio
Materialisierte Sicht aus `AccountBalanceChanged`, `EmpirePayoutRecorded`, `EmpireStageAdvanced`. Keine eigene Aggregationslogik, die nicht 1:1 aus Account/Empire-Events ableitbar ist — sonst entsteht die von dir zurecht befürchtete doppelte Aggregationslogik.

### PerformanceSnapshot
Materialisierte Sicht aus `AllocationClosed` (+ optional periodisch gebatcht). Bezugsebene (Account / Playbook / global) ist ein Filterparameter der Projection, keine strukturelle Unterscheidung.

**Konsequenz für Empire (Klarstellung):** Empire bleibt Domain-Aggregate (Runde 1 korrekt), weil es echte Zustandsübergänge mit Invarianten besitzt (`EmpireStageAdvanced` ist eine geschäftliche Entscheidung, keine reine Ableitung). Portfolio ist keine Domäne, weil es nichts entscheidet, nur zusammenfasst.

---

## 2. Event Catalog (Phase 2, erweitert um Event-Contract)

### 2.1 Pflicht-Envelope für JEDES Event (Runde 2, verbindlich)

```
Event {
  event_type        String, Vergangenheitsform, z.B. "AllocationClosed"
  aggregate_type     z.B. "TradeAllocation"
  aggregate_id        UUID der Aggregate-Instanz, die das Event erzeugt hat
  version              fortlaufende Sequenznummer im Aggregate-Stream (optimistic concurrency)
  timestamp
  source                desktop | mobile | system | scheduled-job
  payload               deterministisch, vollständig selbsterklärend — kein Nachschlagen
                         in anderen Aggregates nötig, um das Event zu interpretieren
}
```

**Regel:** Kein Event ohne eindeutigen `aggregate_type` + `aggregate_id`. Projektionen (Journal, Portfolio) erzeugen daher grundsätzlich keine Events — sie sind reine Konsumenten. Das war in Runde 1 nicht sauber getrennt (`PortfolioSnapshotCreated` hatte keinen klaren Aggregate-Owner) und wird hier korrigiert: Portfolio erzeugt kein Event mehr, sondern ist nur materialisierter Lesezustand.

### 2.2 Event-Tabelle (korrigiert)

| Event | Aggregate Owner | Reagierende Domänen/Projektionen | Veränderte Daten |
|---|---|---|---|
| `SignalGenerated` | Signal | Journal (Projection) | Neues Signal |
| `PlaybookGenerated` / `PlaybookRetired` | Playbook | WeeklyReport | Playbook-Status |
| `AllocationProposed` | TradeAllocation | Risk Gate (sync, vorgelagert) | Neue Allocation (PROPOSED) |
| `AllocationConfirmed` | TradeAllocation | — | Allocation-Status |
| `AllocationOpened` | TradeAllocation | Account, Risk Monitor (async ab jetzt aktiv) | Allocation-Status, Account-Equity (entsteht nur nach 2. Gate-Durchlauf bei `MarkAllocationOpenedCommand`) |
| `AllocationPartialHit` | TradeAllocation | Account, PerformanceSnapshot (Projection) | Allocation-Status, Account-Equity |
| `AllocationBreakEvenActivated` | TradeAllocation | Risk Monitor (Same-Pair-Neubewertung) | Allocation-Status |
| `AllocationClosed` | TradeAllocation | Account (Balance), PerformanceSnapshot, Portfolio (Projections) | Account-Balance |
| `AllocationCancelled` | TradeAllocation | — | Allocation-Status |
| `AccountBalanceChanged` | Account | Empire, Portfolio (Projection), Risk Monitor | Account-Balance |
| `AccountEquityUpdated` | Account | Risk Monitor | Account-Equity |
| `RiskPolicyTriggered` | RiskPolicy | Erzeugt Folge-Command an TradeAllocation/Account | — (löst Command aus, mutiert selbst nichts) |
| `RiskPolicyActivated` / `RiskPolicyRetired` | RiskPolicy | — | Policy-Status |
| `EmpireAccountAdded` | Empire | Portfolio (Projection) | Empire-Accountliste |
| `EmpireStageAdvanceProposed` | Empire (via Risk-Monitor-Trigger) | Desktop-UI (zeigt Bestätigungsdialog) | — (kein Zustandswechsel, reiner Vorschlag) |
| `EmpireStageAdvanced` | Empire (nur nach `ConfirmEmpireStageAdvanceCommand`) | Portfolio (Projection) | Empire-Stage |
| `EmpirePayoutRecorded` | Empire | Portfolio (Projection) | Empire-Historie |
| `JournalNoteAdded` / `JournalNoteEdited` | **JournalNote** (neu) | Journal (Projection) | Notiztext |
| `CalendarEventScheduled` / `Revised` / `Occurred` | CalendarEvent | WeeklyReport, Risk Gate/Monitor, TradeAllocation | CalendarEvent-Status |
| `WeeklyReportGenerated` / `Published` | WeeklyReport | Playbook (Vorschlag) | Report-Status |

`PortfolioSnapshotCreated` und ein eigenständiges `JournalUpdated`/`JournalEntryCreated` **entfallen** gegenüber Runde 1 — sie waren Events ohne sauberen Aggregate-Owner. Portfolio und Journal sind jetzt reine Projektionen ohne eigene Event-Erzeugung.

---

## 3. Risk als Gate + Monitor (Phase 3, korrigiert)

Das ist die zentrale Korrektur aus Runde 2. Risk ist **kein reiner Event-Reagierer**, sondern hat zwei fachlich unterschiedliche Modi:

### 3.1 Risk Gate (synchron, Pre-Trade)

```
Command (z.B. ConfirmAllocationCommand)
        ↓
Risk Gate — synchrone Evaluation aller Policies mit EvaluationMode = SYNC_GATE
        ↓
   Decision:
     - ALLOW
     - ALLOW_WITH_ADJUSTMENT   (z.B. Size 1.0pp → 0.5pp bei Same-Pair)
     - REJECT
        ↓
TradeAllocation-Aggregate führt Command nur bei ALLOW/ALLOW_WITH_ADJUSTMENT aus
        ↓
Event (AllocationConfirmed / AllocationOpened) — trägt die tatsächlich
angewendete, ggf. angepasste Risikogröße im Payload
```

**Harte Regel:** Kein Command, der Kapital exponiert, erreicht das TradeAllocation-Aggregate ohne vorherigen Gate-Durchlauf. Das ist architektonisch erzwungen (Pflichtschicht), nicht optional pro Command.

Policies mit `SYNC_GATE`: Same Pair, Correlation, News Blackout (falls Blackout aktiv → REJECT), Daily Drawdown (falls bereits gerissen → REJECT).

**Bestätigt (Runde 3) — zwei getrennte Gate-Durchläufe, nicht einer:**

```
1. ConfirmAllocationCommand  → Risk Gate prüft GEPLANTES Risiko
                                (Größe, Same-Pair, Correlation, Blackout-Fenster)
                                → AllocationConfirmed

2. MarkAllocationOpenedCommand → Risk Gate prüft ERNEUT, mit echtem Fill
   (trägt tatsächlichen Entry-   (Preis/Spread kann vom Plan abweichen, Account-
    Preis + Fill-Zeit im Payload) Status oder andere offene Risiken können sich
                                    zwischen Confirm und Fill geändert haben)
                                → AllocationOpened (bei ALLOW) oder
                                  AllocationCancelled (bei REJECT trotz Confirm)
```

Grund für den zweiten Durchlauf: Zwischen Planung (Confirm) und tatsächlicher Ausführung (Fill) liegt eine Zeitspanne, in der sich Marktbedingungen oder Account-Zustand geändert haben können — ein einmaliger Check zum Zeitpunkt der Absicht reicht nicht, um die Invariante "kein OPEN ohne aktuell gültiges ALLOW" zu garantieren.

### 3.2 Risk Monitor (asynchron, Event-reaktiv)

Für Zustände, die sich **während** einer bereits offenen Position entwickeln (Equity fällt unter Schwelle, Trailing DD, Profit-Ziel erreicht), bleibt der Event-reaktive Ansatz aus Runde 1 korrekt — hier gibt es keinen "Command davor", auf den man synchron warten könnte, da der Auslöser eine Marktbewegung ist, kein User-Intent.

```
Event-Strom (AccountBalanceChanged, AllocationOpened, CalendarEventOccurred, ...)
        ↓
Risk Monitor wertet Policies mit EvaluationMode = ASYNC_MONITOR aus
        ↓
RiskPolicyTriggered-Event + Folge-Command (z.B. ForceCloseAllocationCommand,
BlockNewAllocationsCommand) an die zuständigen Aggregates
        ↓
Zieldomäne (TradeAllocation/Account) führt Command selbst aus → eigenes Event
```

Policies mit `ASYNC_MONITOR`: Max Drawdown, Trailing DD, Profit Lock, Challenge Target.

**Bestätigt aus Runde 1, präzisiert:** Risk Monitor mutiert nie direkt Account/Allocation-Zustand — sie schickt Commands, die Zieldomäne entscheidet und meldet per eigenem Event zurück. Der Unterschied zu Runde 1: Risk Gate (sync) ist eine **Pflichtschicht vor** Execution, Risk Monitor (async) ist ein **Beobachter nach** Execution-Start. Beide teilen sich dieselben RiskPolicy-Aggregates, unterscheiden sich nur im `EvaluationMode`.

### 3.3 Policy-Tabelle mit EvaluationMode

| Policy | EvaluationMode | Trigger | Aktion |
|---|---|---|---|
| Same Pair | SYNC_GATE | `ConfirmAllocationCommand` | ALLOW_WITH_ADJUSTMENT (0.5pp) |
| Correlation | SYNC_GATE | `ConfirmAllocationCommand` | ALLOW_WITH_ADJUSTMENT / REJECT |
| News Blackout | SYNC_GATE | `ConfirmAllocationCommand` | REJECT (befristet) |
| Daily Drawdown | SYNC_GATE (Prüfung) + ASYNC_MONITOR (Auslösung) | Command-Zeit + `AllocationClosed` | REJECT neuer Commands nach Auslösung |
| Max Drawdown | ASYNC_MONITOR | `AccountBalanceChanged` | ForceCloseAllocationCommand (alle offenen) |
| Trailing DD | ASYNC_MONITOR | `AccountBalanceChanged` | ReduceRiskSizeCommand (nächste Allocation) |
| Profit Lock | ASYNC_MONITOR | `AccountBalanceChanged` | BlockNewAllocationsCommand |
| Challenge Target | ASYNC_MONITOR | `AccountBalanceChanged` | `ProposeEmpireStageAdvanceCommand` (nur Vorschlag — Freigabe zwingend manuell, s. 1.5) |

---

## 4. Architekturdiagramm (korrigiert um Risk Gate + 3-Ebenen-Trennung explizit)

```
┌─────────────┐        ┌─────────────┐
│   Desktop   │        │   Mobile    │      COMMAND LAYER (Intent)
└──────┬──────┘        └──────┬──────┘
       │                       │
       └───────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │   RISK GATE (sync)   │◄── nur für Commands, die Kapital exponieren
         │  ALLOW / ADJUST /    │    (ConfirmAllocationCommand etc.)
         │      REJECT          │
         └──────────┬───────────┘
                    ▼
     ┌─────────────────────────────────────┐
     │           DOMAIN LAYER (Truth)         │
     │  Aggregates mit exklusiver Schreib-     │
     │  autorität über ihren eigenen Zustand:  │
     │                                          │
     │  Signal · Playbook · TradeAllocation ·   │
     │  Account · Empire · RiskPolicy ·         │
     │  CalendarEvent · WeeklyReport ·          │
     │  JournalNote (minimal)                   │
     └───────────────────┬─────────────────────┘
                         │
                      Events
          (envelope: type, aggregate_id, version,
                 timestamp, source, payload)
                         │
                         ▼
              ┌──────────────────┐
              │   Event Log        │  ← Single Source of Truth
              │  (append-only)     │     (Facts, nicht Zustand)
              └────────┬──────────┘
                         │
              ┌──────────┴───────────┐
              ▼                       ▼
   ┌────────────────────┐   ┌─────────────────────┐
   │  RISK MONITOR (async)│   │  PROJECTION LAYER     │
   │  wertet laufende      │   │  (Views)               │
   │  Events gegen ASYNC_  │   │                        │
   │  MONITOR-Policies aus │   │  Journal · Portfolio ·  │
   │  → Folge-Commands      │   │  PerformanceSnapshot    │
   └──────────┬─────────────┘   └──────────┬─────────────┘
              │ (zurück in Command Layer)   │
              ▼                              ▼
   ┌─────────────────────────────────────────────┐
   │              Database                          │
   │   (Event Store + Projection-Tabellen)          │
   └───────────────────────┬─────────────────────────┘
                            │
              ┌──────────────┴───────────────┐
              ▼                               ▼
        ┌───────────┐                  ┌───────────┐
        │  Desktop   │                  │  Mobile   │
        │ liest       │                  │ liest      │
        │ Projektionen │                 │ Projektionen│
        └───────────┘                  └───────────┘
```

Der Risk Gate ist bewusst **im Command-Pfad selbst gezeichnet, nicht als Domäne daneben** — das ist genau die Korrektur, die du eingefordert hast: Execution kann strukturell nicht ohne Risk-Evaluation starten, weil der Pfad gar nicht anders existiert.

---

## 5. Aggregate Map (neu, Runde 2)

Wer darf was verändern — eine Zeile pro Aggregate.

| Aggregate | Mutiert exklusiv | Akzeptiert Commands | Kern-Invarianten |
|---|---|---|---|
| **Signal** | eigenen Status (GENERATED/EXPIRED) | `GenerateSignalsCommand` | Immutable nach Erzeugung |
| **Playbook** | eigenen Status | `CreatePlaybookCommand`, `RetirePlaybookCommand` | Kein Status-Sprung DRAFT→RETIRED ohne ACTIVE |
| **TradeAllocation** | eigenen Lifecycle-Status, referenzierte Risikogröße | `ProposeAllocationCommand`, `ConfirmAllocationCommand`, `MarkAllocationOpenedCommand`, `ForceCloseAllocationCommand`, `CancelAllocationCommand` | Kein Übergang Richtung CONFIRMED/OPEN ohne je eigenen Risk-Gate-ALLOW (zwei unabhängige Durchläufe); genau 1 Account-Referenz |
| **Account** | Balance, Equity, Status (ACTIVE/BREACHED/...) | `AdjustBalanceCommand` (nur system-intern via Allocation-Reaktion), `MarkAccountBreachedCommand` | Balance ändert sich nur durch `AllocationClosed`-Reaktion, nie direkt von außen |
| **Empire** | Accountliste, Stage, Payout-Historie | `AddAccountToEmpireCommand`, `ProposeEmpireStageAdvanceCommand`, `ConfirmEmpireStageAdvanceCommand`, `RecordPayoutCommand` | Stage-Progression ist monoton UND erfordert immer manuelle Bestätigung — kein automatischer Sprung, auch nicht bei eindeutigem Trigger |
| **RiskPolicy** | eigenen Status, Konfiguration | `DefinePolicyCommand`, `ActivatePolicyCommand`, `RetirePolicyCommand` | Jede Policy hat genau einen `EvaluationMode` |
| **CalendarEvent** | eigenen Status | `SyncCalendarCommand` | Nur `SCHEDULED → REVISED* → OCCURRED`, keine Rückwärtsübergänge |
| **WeeklyReport** | eigenen Status, Inhalt | `GenerateWeeklyReportCommand`, `PublishWeeklyReportCommand` | Kein PUBLISHED ohne vorheriges GENERATED |
| **JournalNote** | eigenen Text | `AddJournalNoteCommand`, `EditJournalNoteCommand` | Referenziert genau 1 Allocation ODER 1 Signal, nie beides, nie 0 |

**Was hier fehlt, ist Absicht:** Journal, Portfolio, PerformanceSnapshot tauchen nicht auf — sie mutieren nichts, sie lesen nur (Abschnitt 1b).

---

## 6. Command → Decision → Event Matrix (neu, Runde 2)

| Command | Sender | Risk Gate? | Decision-Optionen | Resultierendes Event |
|---|---|---|---|---|
| `GenerateSignalsCommand` | Desktop (Research-Job) | nein | — | `SignalGenerated` |
| `ProposeAllocationCommand` | Desktop/System | nein (Proposal ist noch kein Kapitaleinsatz) | — | `AllocationProposed` |
| `ConfirmAllocationCommand` | Desktop/Mobile | **ja, SYNC_GATE (Durchlauf 1 — geplantes Risiko)** | ALLOW / ALLOW_WITH_ADJUSTMENT / REJECT | `AllocationConfirmed` (bei ALLOW/ADJUST) oder keines (bei REJECT — Command schlägt fehl) |
| `MarkAllocationOpenedCommand` (Broker-Fill-Bestätigung, trägt echten Entry) | System | **ja, SYNC_GATE (Durchlauf 2 — Ist-Zustand bei Fill)** | ALLOW / REJECT | `AllocationOpened` (bei ALLOW) oder `AllocationCancelled` (bei REJECT trotz vorherigem Confirm) |
| `ForceCloseAllocationCommand` | Risk Monitor (async) | nein (ist selbst die Konsequenz einer Risk-Entscheidung) | — | `AllocationClosed` |
| `CancelAllocationCommand` | Desktop/Mobile/Risk Gate (bei REJECT) | nein | — | `AllocationCancelled` |
| `AddJournalNoteCommand` | Desktop/Mobile | nein | — | `JournalNoteAdded` (nur subjektiver Kommentar/Screenshot/Lessons Learned — nie Trade-Fakten) |
| `ProposeEmpireStageAdvanceCommand` | Risk Monitor (async, aus Challenge-Target-Trigger) | nein | Erzeugt nur einen Vorschlag, keine Zustandsänderung | `EmpireStageAdvanceProposed` |
| `ConfirmEmpireStageAdvanceCommand` | **Desktop (manuelle Bestätigung durch User, Pflicht)** | nein | — | `EmpireStageAdvanced` |
| `RecordPayoutCommand` | Desktop (manuell) | nein | — | `EmpirePayoutRecorded` |
| `SyncCalendarCommand` | System (Scheduler) | nein | — | `CalendarEventScheduled/Revised/Occurred` |
| `GenerateWeeklyReportCommand` | Desktop | nein | — | `WeeklyReportGenerated` |
| `PublishWeeklyReportCommand` | Desktop (redaktionelle Freigabe) | nein | — | `WeeklyReportPublished` |
| `DefinePolicyCommand` | Desktop | nein | — | `RiskPolicyActivated` |

**Lesehinweis:** Nur Commands, die tatsächlich Kapital exponieren (`ConfirmAllocationCommand`, `OpenAllocationCommand`), durchlaufen den synchronen Gate. Alle anderen Commands sind entweder folgenlos für Risiko (Journal, Reports, Calendar) oder bereits das Ergebnis einer Risk-Entscheidung (`ForceCloseAllocationCommand` kommt vom Monitor selbst).

---

## 7. Roadmap (bestätigt/präzisiert nach deiner Entscheidung: Execution vor Research)

**Stufe 1 — Event Store + Account/Empire-Aggregates**
Fundament. Empire-Modul (64 Tests) wird auf Event-Erzeugung gehoben, nicht neu gebaut.

**Stufe 2 — TradeAllocation + Risk Gate (SYNC_GATE-Policies) + Risk Monitor (ASYNC_MONITOR-Policies)**
Beide Risk-Modi kommen **gemeinsam** mit TradeAllocation, nicht nachgelagert — sonst existiert kurzzeitig ein Execution-Pfad ohne Gate, was architektonisch nicht erlaubt sein soll (Punkt 1 deiner Korrektur).

**Stufe 3 — JournalNote-Aggregate + Journal-Projection**
Jetzt möglich, weil `AllocationOpened/Closed`-Events aus Stufe 2 zuverlässig fließen. Projection zuerst simpel (reine chronologische Sicht), Live-/Signal-Filterung ist reine Query-Logik obenauf.

**Stufe 4 — Research: Signal + Playbook + Calendar**
Bewusst nach Execution (deine Entscheidung, Punkt 2): Signal-Generierung dockt an ein bereits stabiles Event-Fundament an, statt dass Execution später gegen ein bewegliches Research-Modell nachgezogen wird.

**Stufe 5 — Portfolio & PerformanceSnapshot (reine Projektionen)**
Unverändert zuletzt vor Mobile — jetzt explizit als Query-Layer ohne eigene Events.

**Stufe 6 — WeeklyReport-Integration**
Unverändert.

**Stufe 7 — Mobile Companion**
Unverändert, jetzt mit der Präzisierung: Mobile sendet u.a. `ConfirmAllocationCommand` und `AddJournalNoteCommand` — beide bereits in der Command-Matrix (Abschnitt 6) definiert, keine Sonderfälle nötig.

---

## Phase 1 — EINGEFROREN (Runde 3)

Alle offenen Punkte sind entschieden:

1. **JournalNote** ist ausschließlich subjektiver Kommentar-Layer (Text, Screenshots, Emotionen, Lessons Learned) — führt niemals Trade-Fakten (Ergebnis, R-Multiple, Status, Preise). Harte Daten kommen ausschließlich aus Events und werden nur in der Journal-Projection dazugemischt.
2. **Risk Gate läuft zweimal**, unabhängig voneinander: bei `ConfirmAllocationCommand` (geplantes Risiko) und bei `MarkAllocationOpenedCommand` (Ist-Zustand bei tatsächlichem Fill) — weil sich Preis, Spread, Account-Status oder offene Risiken zwischen Plan und Ausführung ändern können.
3. **Empire-Stage-Progression hat keine Automatik**: Risk Monitor erzeugt nur `EmpireStageAdvanceProposed` (z.B. "Phase 1 vermutlich bestanden – bestätigen?"), der tatsächliche `EmpireStageAdvanced`-Übergang erfolgt ausschließlich durch manuelle `ConfirmEmpireStageAdvanceCommand`-Bestätigung des Users.

Damit ist die fachliche Architektur (Aggregates, Projections, Event-Contract, Risk Gate/Monitor, Command→Decision→Event-Matrix, Roadmap) als Grundlage für Phase 2 eingefroren.

---

## Phase 2 (nächster Schritt, separat)

Ableitung aus dieser eingefrorenen Architektur:
- Datenbankschema (Event Store + Projection-Tabellen)
- Backend-Services pro Aggregate (Command-Handler + Invarianten-Durchsetzung)
- Risk Gate/Monitor als eigenständiger Service oder In-Process-Layer
- API-Oberfläche für Desktop/Mobile (Command-Endpunkte + Projection-Reads)

Wird in einer eigenen Session behandelt, sobald du grünes Licht gibst.
