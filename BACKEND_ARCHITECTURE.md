# Trading Operating System — Backend Architecture (Phase 2)

_Status: **EINGEFROREN** — Grundlage für die Implementierungsphase_
_Basis: `DOMAIN_ARCHITECTURE.md` (eingefroren, Runde 3) — wird nicht verändert, nur abgeleitet._
_Scope: Technisches Fundament. Kein UI, kein React, kein Mobile-Client-Code._
_Leitlinie: Optimiere nicht für das heutige Feature, sondern für das nächste unbekannte Feature._

---

## 0. Entschiedene Konflikte / Lücken aus Phase 1

Drei Punkte waren technisch unterspezifiziert und sind jetzt mit dir final entschieden (nicht mehr offen):

**Entscheidung 1 — `MarkAllocationOpenedCommand` wird in v1 manuell ausgelöst.**
Kein Broker-Integration in v1. Workflow: Playbook → Allocation bestätigt (`ConfirmAllocationCommand`, Gate 1) → User eröffnet manuell beim Broker → User trägt im Terminal/Mobile Entry-Preis + Fill-Zeit ein und bestätigt ("Opened") → `MarkAllocationOpenedCommand` (Gate 2). Sender ist damit **Desktop/Mobile**, nicht `System` — die Command-Matrix (Abschnitt 6 im Domain-Dokument) wird dadurch nicht falsch, nur der tatsächliche Absender ist ein menschlicher Client statt eines automatisierten Systems. Eine spätere `BrokerIntegration`-Komponente kann denselben Command automatisch auslösen (Adapter, kein neues Aggregate) — Domain Layer und Event-Contract bleiben dabei unverändert.

**Entscheidung 2 — RiskPolicy-Scope als Ebenen-Hierarchie mit Override.**
Scope-Typen: `GLOBAL → USER → EMPIRE → PROP_FIRM → ACCOUNT → SIGNAL → ALLOCATION` (aufsteigende Spezifität). `PROP_FIRM` ist eine **Template-Ebene** (z.B. FundingPips-Standardregeln), `ACCOUNT` kann eine Policy mit demselben `policy_key` überschreiben. Praktisch wichtigste Ebene: `ACCOUNT`, weil Prop-Firm-Regeln pro Account gelten. Technisches Modell in 2.1.

**Entscheidung 3 — `user_id` ab Tag 1, aber ohne Team-/Org-Komplexität.**
Jeder Account, jedes Signal, jeder Report gehört genau einem User (`User → owns Accounts, Signals, Reports`; Allocations erben Ownership transitiv über ihren Account, keine eigene Spalte nötig — sonst zwei Wahrheiten). Kein Sharing, keine Teams, keine Orgs in v1 — nur das einfache Owner-Modell, damit spätere Mehrbenutzerfähigkeit nicht nachgerüstet werden muss.

---

## 2.1 Persistence Layer

### Grundsatzentscheidung: PostgreSQL als alleiniger Speicher für Event Store + Projektionen

**Alternativen kurz verglichen:**

| Option | Für | Gegen |
|---|---|---|
| Dediziertes Event-Store-Produkt (EventStoreDB, Kafka) | Für sehr hohen Event-Durchsatz gebaut, native Subscriptions | Zusätzliche Infrastruktur/Ops-Last für ein Ein-Personen-Projekt; kein Mehrwert bei aktuellem Volumen (Dutzende bis Hunderte Events/Tag) |
| Getrennte DBs für Event Store vs. Projektionen | Saubere Trennung von Schreib-/Lesepfad | Verteilte Transaktion nötig, um Event-Append + State-Update atomar zu halten — genau das braucht der Risk Gate (Abschnitt 2.4), um synchron korrekt zu entscheiden |
| **PostgreSQL, ein Cluster, zwei Schemas** | ACID-Transaktion über Event-Append + State-Update in einem Commit; ein Ops-Surface; JSONB deckt Payload-Flexibilität ab; Migration zu Kafka/EventStoreDB später möglich, da Event-Contract storage-agnostisch bleibt (LISTEN/NOTIFY oder logische Replikation als Brücke) | Horizontale Skalierung irgendwann begrenzt — aber erst relevant bei einem Volumen, das für dieses System auf absehbare Zeit nicht erreicht wird |

**Entscheidung:** PostgreSQL, zwei Schemas: `core` (Event Store + Aggregate-State-Tabellen, kritisch, nie manuell anfassen) und `projections` (Journal/Portfolio/Performance, jederzeit aus `core` neu berechenbar, daher weniger schützenswert — andere Backup-/Zugriffspolitik möglich).

### Event Store (`core.event_store`)

```sql
CREATE TABLE core.event_store (
    global_seq       BIGSERIAL PRIMARY KEY,           -- globale Ordnung über alle Aggregates hinweg
    event_id         UUID NOT NULL DEFAULT gen_random_uuid(),
    aggregate_type   TEXT NOT NULL,                    -- 'TradeAllocation', 'Account', ...
    aggregate_id     UUID NOT NULL,
    version          INTEGER NOT NULL,                 -- fortlaufend pro Aggregate-Stream, startet bei 1
    event_type       TEXT NOT NULL,                    -- 'AllocationClosed', ...
    schema_version   INTEGER NOT NULL DEFAULT 1,        -- für Payload-Evolution, s. 2.3
    payload          JSONB NOT NULL,
    source           TEXT NOT NULL,                    -- 'desktop' | 'mobile' | 'system' | 'scheduled-job'
    device_id        UUID NULL REFERENCES core.devices(id),
    correlation_id   UUID NOT NULL,                     -- verknüpft alle Events einer Command-Verarbeitung
    causation_id     UUID NULL,                          -- welches Event/Command hat dieses ausgelöst
    occurred_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    recorded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (aggregate_type, aggregate_id, version)        -- erzwingt Optimistic Concurrency pro Stream
);

CREATE INDEX idx_event_store_aggregate ON core.event_store (aggregate_type, aggregate_id, version);
CREATE INDEX idx_event_store_type      ON core.event_store (event_type);
CREATE INDEX idx_event_store_occurred  ON core.event_store (occurred_at);
```

**Begründung der Kernentscheidungen:**
- `UNIQUE (aggregate_type, aggregate_id, version)` ist der zentrale Konsistenz-Mechanismus: Zwei konkurrierende Commands auf demselben Aggregate (z.B. Desktop und Mobile bestätigen gleichzeitig dieselbe Allocation) können nicht beide Version N erzeugen — der zweite INSERT schlägt fehl, das ist die technische Umsetzung von "kein Lost Update", ohne verteiltes Locking.
- **Immutability erzwungen auf DB-Ebene**, nicht nur per Konvention: keine `UPDATE`/`DELETE`-Rechte auf `core.event_store` für die Anwendungsrolle (nur `INSERT`, `SELECT`). Das verhindert, dass ein Bug jemals Historie überschreibt — kritisch für ein System, das seine eigenen historischen Kennzahlen (Sharpe, Hit-Rate) glaubwürdig halten muss.
- `correlation_id`/`causation_id` sind Pflicht für Nachvollziehbarkeit: "Welcher User-Klick hat letztlich zu diesem `AllocationClosed` geführt" — unverzichtbar, sobald Risk Monitor Folge-Commands auslöst (Event → Command → Event-Kette).
- `global_seq` (statt nur `occurred_at`) garantiert eine total geordnete Wiedergabe für Replay — Zeitstempel allein sind bei Systemuhren-Ungenauigkeit nicht verlässlich genug.

### Aggregate-State-Tabellen (`core` Schema)

**Entscheidung: Hybrid statt Pure-Event-Replay.** Reines Event-Sourcing würde bei jedem Command den kompletten Event-Stream eines Aggregates neu abspielen, um den aktuellen Zustand zu ermitteln. Das ist für den **synchronen Risk Gate** (muss schnell antworten) und für langlebige Aggregates (eine `TradeAllocation` über Jahre Historie) ungeeignet. Stattdessen: Jedes Aggregate hat eine **transaktional mitgeführte State-Tabelle**, die in **derselben DB-Transaktion** wie der Event-Append geschrieben wird (kein Eventual-Consistency-Risiko für den Schreibpfad selbst — nur die Projektionen in Abschnitt 1b sind eventually consistent).

```sql
CREATE TABLE core.accounts (
    id                    UUID PRIMARY KEY,
    user_id               UUID NOT NULL REFERENCES core.users(id),   -- Konflikt 3, additiv
    empire_id             UUID NULL REFERENCES core.empires(id),
    prop_firm_template_id TEXT NULL REFERENCES core.prop_firm_templates(id),  -- Risk-Scope-Ebene PROP_FIRM
    account_type          TEXT NOT NULL CHECK (account_type IN ('LIVE','PROP_FIRM')),
    stage                 TEXT NOT NULL DEFAULT 'ACTIVE',              -- CHALLENGE/VERIFICATION/FUNDED für Prop-Firm
    status                TEXT NOT NULL CHECK (status IN ('ACTIVE','BREACHED','PASSED','FUNDED','CLOSED','ARCHIVED')),
    balance               NUMERIC(14,2) NOT NULL,
    equity                NUMERIC(14,2) NOT NULL,
    risk_max_dd_pct        NUMERIC(5,2) NOT NULL,
    risk_daily_dd_pct       NUMERIC(5,2) NOT NULL,
    risk_per_trade_pct      NUMERIC(5,2) NOT NULL,
    version               INTEGER NOT NULL DEFAULT 0,                 -- muss mit event_store.version synchron sein
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE core.empires (
    id            UUID PRIMARY KEY,
    user_id       UUID NOT NULL REFERENCES core.users(id),
    name          TEXT NOT NULL,
    version       INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE core.trade_allocations (
    id                   UUID PRIMARY KEY,
    account_id           UUID NOT NULL REFERENCES core.accounts(id),
    signal_id            UUID NULL REFERENCES core.signals(id),
    playbook_id          UUID NULL REFERENCES core.playbooks(id),
    pair                 TEXT NOT NULL,
    direction            TEXT NOT NULL CHECK (direction IN ('LONG','SHORT')),
    status               TEXT NOT NULL CHECK (status IN
                          ('PROPOSED','CONFIRMED','OPEN','PARTIAL_HIT','BREAK_EVEN','CLOSED','CANCELLED')),
    planned_risk_pct      NUMERIC(5,2) NOT NULL,
    applied_risk_pct      NUMERIC(5,2) NULL,                          -- vom Risk Gate ggf. angepasst
    entry_price_planned   NUMERIC(12,5) NULL,
    fill_price            NUMERIC(12,5) NULL,
    fill_time             TIMESTAMPTZ NULL,
    sl_price              NUMERIC(12,5) NULL,
    tp_price              NUMERIC(12,5) NULL,
    closed_at             TIMESTAMPTZ NULL,
    close_reason          TEXT NULL CHECK (close_reason IN ('SL','TP','TIME_EXIT','FORCE_CLOSE','CANCELLED') OR close_reason IS NULL),
    realized_r            NUMERIC(6,3) NULL,
    version               INTEGER NOT NULL DEFAULT 0,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_allocations_account_status ON core.trade_allocations (account_id, status);
CREATE INDEX idx_allocations_pair_status    ON core.trade_allocations (pair, status);  -- Same-Pair-Policy braucht das
CREATE INDEX idx_allocations_closed_at      ON core.trade_allocations (closed_at);

CREATE TABLE core.signals (
    id                   UUID PRIMARY KEY,
    user_id              UUID NOT NULL REFERENCES core.users(id),   -- Entscheidung 3: Owner ab Tag 1
    pair                 TEXT NOT NULL,
    direction            TEXT NOT NULL,
    confidence_rank       TEXT NULL,                -- 'TOP1','TOP2','SMART3', ...
    model_ensemble_version TEXT NOT NULL,
    status               TEXT NOT NULL CHECK (status IN ('GENERATED','EXPIRED')),
    generated_at          TIMESTAMPTZ NOT NULL,
    valid_until           TIMESTAMPTZ NOT NULL,
    version               INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE core.playbooks (
    id            UUID PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('DRAFT','ACTIVE','RETIRED')),
    version       INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE core.prop_firm_templates (      -- Template-Ebene, z.B. 'fundingpips-standard'
    id     TEXT PRIMARY KEY,
    name   TEXT NOT NULL
);

-- accounts.prop_firm_template_id (nachgetragen, s. Account-Tabelle oben) verknüpft
-- einen Account optional mit genau einem Template.

CREATE TABLE core.risk_policies (
    id                TEXT PRIMARY KEY,             -- sprechender Key, z.B. 'same-pair-account-xyz'
    policy_key        TEXT NOT NULL,                 -- fachliche Identität, z.B. 'max-drawdown' —
                                                        -- gleich über alle Scope-Ebenen hinweg, macht Overrides möglich
    name              TEXT NOT NULL,
    description       TEXT NOT NULL,
    evaluation_mode    TEXT NOT NULL CHECK (evaluation_mode IN ('SYNC_GATE','ASYNC_MONITOR')),
    trigger_events     TEXT[] NOT NULL,               -- Liste von event_type / command_type Strings
    priority          INTEGER NOT NULL,
    action_type       TEXT NOT NULL CHECK (action_type IN
                       ('ALLOW','ALLOW_WITH_ADJUSTMENT','REJECT','FORCE_CLOSE',
                        'REDUCE_RISK_SIZE','BLOCK_NEW_ALLOCATIONS','LOCK_PROFIT','PROPOSE_STAGE_ADVANCE')),
    scope_type        TEXT NOT NULL CHECK (scope_type IN
                       ('GLOBAL','USER','EMPIRE','PROP_FIRM','ACCOUNT','SIGNAL','ALLOCATION')),
    scope_id          TEXT NULL,                       -- NULL nur bei GLOBAL; sonst FK-artige Referenz
                                                          -- auf user_id/empire_id/prop_firm_template_id/
                                                          -- account_id/signal_id/allocation_id je nach scope_type
                                                          -- (kein echter FK, da Zieltabelle variiert — Prüfung im
                                                          -- Repository, nicht in SQL)
    status            TEXT NOT NULL CHECK (status IN ('DEFINED','ACTIVE','SUSPENDED','RETIRED')),
    version           INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT chk_global_has_no_scope_id CHECK (
        (scope_type = 'GLOBAL' AND scope_id IS NULL) OR (scope_type <> 'GLOBAL' AND scope_id IS NOT NULL)
    )
);

CREATE INDEX idx_risk_policies_scope ON core.risk_policies (scope_type, scope_id);
CREATE INDEX idx_risk_policies_key   ON core.risk_policies (policy_key);

CREATE TABLE core.calendar_events (
    id             UUID PRIMARY KEY,
    title          TEXT NOT NULL,
    currency       TEXT NOT NULL,
    impact         TEXT NOT NULL CHECK (impact IN ('LOW','MEDIUM','HIGH')),
    status         TEXT NOT NULL CHECK (status IN ('SCHEDULED','REVISED','OCCURRED')),
    scheduled_at   TIMESTAMPTZ NOT NULL,
    occurred_at    TIMESTAMPTZ NULL,
    version        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE core.weekly_reports (
    id             UUID PRIMARY KEY,
    user_id        UUID NOT NULL REFERENCES core.users(id),   -- Entscheidung 3: Owner ab Tag 1
    period_start   DATE NOT NULL,
    period_end     DATE NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('DRAFT','GENERATED','PUBLISHED','ARCHIVED')),
    content_ref    TEXT NULL,          -- Pfad/URL zum generierten PDF, keine großen Inhalte in der Zeile
    published_at   TIMESTAMPTZ NULL,
    version        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE core.journal_notes (
    id                    UUID PRIMARY KEY,
    related_allocation_id  UUID NULL REFERENCES core.trade_allocations(id),
    related_signal_id      UUID NULL REFERENCES core.signals(id),
    text                  TEXT NOT NULL,
    attachments           JSONB NOT NULL DEFAULT '[]',   -- Screenshot-URLs, rein deskriptiv
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    edited_at             TIMESTAMPTZ NULL,
    version               INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT chk_exactly_one_reference CHECK (
        (related_allocation_id IS NOT NULL)::int + (related_signal_id IS NOT NULL)::int = 1
    )
);
```

**Wichtige strukturelle Entscheidung (setzt Phase-1-Regel technisch durch):** `journal_notes` besitzt **keine** Spalten für Ergebnis/R-Multiple/Preise/Status — bewusst, damit "JournalNote speichert niemals Trade-Fakten" nicht nur Konvention, sondern durch fehlende Spalten unmöglich ist.

**Ownership-Modell (Entscheidung 3):** `trade_allocations` trägt bewusst **kein eigenes** `user_id` — Ownership ergibt sich transitiv über `account_id → accounts.user_id`. Eine redundante Spalte wäre eine zweite Wahrheit, die bei einem (aktuell nicht vorgesehenen) Account-Transfer stillschweigend veralten könnte. `playbooks` bleibt bewusst ohne `user_id` — Playbooks sind aktuell geteiltes Referenzwissen (keine Team-Logik nötig, da nur ein User), nicht individuell zugeordnetes Eigentum; sollte das später relevant werden, ist es eine additive Spalte, keine Umstrukturierung.

**Weitere Konsistenz-Durchsetzung auf DB-Ebene:**
- `core.accounts.balance`/`equity`: `UPDATE`-Recht in der Datenbank-Rolle nur für den Account-Aggregate-Service, nicht für andere Service-Rollen — technische Umsetzung von "Balance ändert sich nur durch `AllocationClosed`-Reaktion, nie direkt von außen".
- `chk_exactly_one_reference` erzwingt die JournalNote-Invariante direkt als CHECK-Constraint, nicht nur in Anwendungscode.

### Projections (`projections` Schema)

```sql
CREATE TABLE projections.journal_view (
    id                    UUID PRIMARY KEY,
    track                 TEXT NOT NULL CHECK (track IN ('LIVE','SIGNAL')),
    allocation_id          UUID NULL,
    signal_id             UUID NULL,
    entry_summary          JSONB NOT NULL,      -- denormalisiert: Pair, Richtung, Ergebnis, Zeitpunkt
    notes                 JSONB NOT NULL DEFAULT '[]',  -- eingebettete JournalNote-Texte
    occurred_at            TIMESTAMPTZ NOT NULL
);

CREATE TABLE projections.portfolio_snapshots (
    id                UUID PRIMARY KEY,
    as_of             TIMESTAMPTZ NOT NULL,
    total_balance_eur  NUMERIC(14,2) NOT NULL,
    breakdown         JSONB NOT NULL             -- pro Account/Empire
);

CREATE TABLE projections.performance_snapshots (
    id             UUID PRIMARY KEY,
    scope_type     TEXT NOT NULL CHECK (scope_type IN ('ACCOUNT','PLAYBOOK','GLOBAL')),
    scope_id       UUID NULL,
    period_start   DATE NOT NULL,
    period_end     DATE NOT NULL,
    win_rate       NUMERIC(5,2) NOT NULL,
    sharpe         NUMERIC(6,3) NOT NULL,
    max_dd_pct      NUMERIC(5,2) NOT NULL,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE projections.checkpoints (       -- Replay-Fähigkeit, s. 2.3
    projection_name   TEXT PRIMARY KEY,
    last_processed_seq BIGINT NOT NULL DEFAULT 0
);
```

Projektionen tragen **keine Foreign Keys zu `core`** — bewusst entkoppelt, weil sie jederzeit gelöscht und aus `core.event_store` neu aufgebaut werden können (Rebuild = `TRUNCATE` + Replay). FK-Constraints würden diese Wegwerfbarkeit erschweren.

### Migration-Struktur

**Werkzeug:** Alembic (Python-nativ, passt zum bestehenden `src/`-Stack des Projekts).

```
migrations/
  versions/
    0001_core_event_store.py
    0002_core_users_devices_sessions.py
    0003_core_accounts_empires.py
    0004_core_trade_allocations.py
    0005_core_signals_playbooks.py
    0006_core_risk_policies.py
    0007_core_calendar_events.py
    0008_core_weekly_reports.py
    0009_core_journal_notes.py
    0010_projections_schema.py
    ...
  env.py
```

**Regel:** Migrationen sind additiv-only in `core.event_store` (nie ein `ALTER TABLE ... DROP COLUMN` auf Event-Daten) — Schema-Änderungen an Event-Payloads laufen über Payload-Versionierung (2.3), nicht über SQL-Migrationen der JSONB-Struktur. Migrationen an State-Tabellen und Projektionen sind normal (additive Spalten bevorzugt, Breaking Changes nur mit Backfill-Migration + Tests).

---

## 2.2 Domain Layer — Struktur

```
domain/
├── shared/
│   ├── aggregate_root.py        # Basisklasse: version, uncommitted_events, apply()
│   ├── event_envelope.py        # Envelope-Contract (2.1 Event-Tabelle)
│   ├── command.py               # Command-Basisklasse (command_id, correlation_id, issued_by)
│   ├── unit_of_work.py          # Transaktionale Klammer: Event-Append + State-Update atomar
│   ├── event_bus.py             # Interface: publish(event), subscribe(event_type, handler)
│   └── repository.py            # Interface: load(id) -> Aggregate, save(aggregate, expected_version)
│
├── account/
│   ├── account.py                # Aggregate: Balance/Equity/Status, Invarianten
│   ├── account_repository.py
│   ├── events.py                 # AccountBalanceChanged, AccountEquityUpdated
│   └── commands.py               # AdjustBalanceCommand (intern), MarkAccountBreachedCommand
│
├── empire/
│   ├── empire.py                  # Aggregate: Accountliste, Stage, Payout-Historie
│   ├── empire_repository.py
│   ├── events.py                  # EmpireAccountAdded, EmpireStageAdvanceProposed, EmpireStageAdvanced, EmpirePayoutRecorded
│   └── commands.py                # AddAccountToEmpireCommand, ProposeEmpireStageAdvanceCommand,
│                                    # ConfirmEmpireStageAdvanceCommand, RecordPayoutCommand
│
├── allocation/                      # Execution-Context (TradeAllocation)
│   ├── trade_allocation.py          # Aggregate: Lifecycle-State-Machine, Invarianten
│   ├── allocation_repository.py
│   ├── allocation_lifecycle_service.py  # orchestriert Propose→Confirm→Open→Close,
│   │                                       ruft RiskGateService synchron auf (kein direkter Aufruf
│   │                                       der Risk-Policies aus dem Aggregate selbst — Trennung
│   │                                       von Invariante (Aggregate) und Entscheidung (Risk))
│   ├── events.py                    # AllocationProposed/Confirmed/Opened/PartialHit/
│   │                                  # BreakEvenActivated/Closed/Cancelled
│   └── commands.py                  # ProposeAllocationCommand, ConfirmAllocationCommand,
│                                       # MarkAllocationOpenedCommand, ForceCloseAllocationCommand,
│                                       # CancelAllocationCommand
│
├── risk/
│   ├── risk_policy.py                # Aggregate: Konfiguration, Status, EvaluationMode
│   ├── risk_policy_repository.py
│   ├── risk_gate_service.py           # SYNCHRON, in-process aufgerufen von allocation_lifecycle_service
│   │                                    # evaluiert alle ACTIVE Policies mit EvaluationMode=SYNC_GATE,
│   │                                    # gibt Decision zurück (ALLOW/ADJUST/REJECT), MUTIERT NICHTS
│   ├── policy_scope_resolver.py        # sammelt für einen Evaluationskontext (Account/Signal/Allocation)
│   │                                    # alle anwendbaren Policies über die Scope-Hierarchie
│   │                                    # GLOBAL→USER→EMPIRE→PROP_FIRM→ACCOUNT→SIGNAL→ALLOCATION,
│   │                                    # löst Overrides pro policy_key auf (höchste Spezifität gewinnt)
│   ├── risk_monitor_service.py         # ASYNCHRON, Event-Bus-Subscriber, evaluiert ASYNC_MONITOR-
│   │                                    # Policies gegen eingehende Events, erzeugt Folge-Commands
│   ├── policies/                       # eine Datei pro konkreter Policy, gemeinsames Interface
│   │   ├── same_pair_policy.py
│   │   ├── correlation_policy.py
│   │   ├── news_blackout_policy.py
│   │   ├── daily_drawdown_policy.py
│   │   ├── max_drawdown_policy.py
│   │   ├── trailing_dd_policy.py
│   │   ├── profit_lock_policy.py
│   │   └── challenge_target_policy.py
│   └── events.py                       # RiskPolicyTriggered, RiskPolicyActivated, RiskPolicyRetired
│
├── journal/
│   ├── journal_note.py                 # Aggregate: minimal, nur Text + genau 1 Referenz
│   ├── journal_note_repository.py
│   ├── events.py                       # JournalNoteAdded, JournalNoteEdited
│   └── commands.py                     # AddJournalNoteCommand, EditJournalNoteCommand
│
├── research/
│   ├── signal.py                       # Aggregate
│   ├── playbook.py                     # Aggregate
│   ├── signal_repository.py / playbook_repository.py
│   ├── signal_generation_service.py     # orchestriert bestehende ML-Pipeline, verpackt Resultat
│   │                                      # als GenerateSignalsCommand-Ausführung
│   ├── events.py                        # SignalGenerated, PlaybookGenerated, PlaybookRetired
│   └── commands.py                      # GenerateSignalsCommand, CreatePlaybookCommand, RetirePlaybookCommand
│
├── calendar/
│   ├── calendar_event.py                 # Aggregate
│   ├── calendar_event_repository.py
│   ├── calendar_sync_service.py           # kapselt externen Scraper (bestehender calendar_fetcher.py)
│   ├── events.py                          # CalendarEventScheduled/Revised/Occurred
│   └── commands.py                        # SyncCalendarCommand
│
└── weekly_report/
    ├── weekly_report.py                    # Aggregate
    ├── weekly_report_repository.py
    ├── events.py                           # WeeklyReportGenerated, WeeklyReportPublished
    └── commands.py                         # GenerateWeeklyReportCommand, PublishWeeklyReportCommand
```

**Verantwortlichkeiten pro Bausteintyp (gilt kontextübergreifend):**

- **Aggregate:** Einzige Instanz mit Schreibautorität über den eigenen Zustand. Prüft Invarianten synchron beim Anwenden eines Commands, erzeugt Events als Ergebnis. Kennt keine anderen Aggregates direkt (nur über IDs).
- **Repository:** Lädt/speichert ein Aggregate über Event Store + State-Tabelle, erzwingt Optimistic Concurrency (`expected_version`). Einziger Ort, der SQL kennt — Aggregate selbst ist persistenzunwissend.
- **Service (dort wo vorhanden):** Orchestriert einen Ablauf, der mehr als ein Aggregate/eine Entscheidung braucht (z.B. Allocation-Lifecycle ruft Risk Gate auf), aber besitzt selbst keinen Zustand und keine Invarianten — reine Koordination.
- **Policy (nur Risk):** Reine Entscheidungsfunktion `evaluate(context) -> Decision`, zustandslos, testbar ohne DB.
- **Events/Commands:** Reine Datenverträge (DTOs), keine Logik.

**Projections liegen bewusst außerhalb von `domain/`** (in `projections/` auf Anwendungsebene) — sie sind Konsumenten des Event-Bus, keine Domain-Bausteine, das spiegelt die 3-Ebenen-Trennung aus Phase 1 auch im Code wider.

---

## 2.3 Event Infrastructure

### Event Store
Wie in 2.1 — `core.event_store`, einziger Schreibpfad über `UnitOfWork.append()`.

### Event Publishing

**Muster: Transactional Outbox, Start einfach, Interface zukunftsfest.**

```
UnitOfWork.commit():
    BEGIN
      INSERT INTO core.event_store (...)   -- die Wahrheit
      UPDATE core.<aggregate_table> ...     -- State-Tabelle, gleiche Transaktion
    COMMIT
    -- NACH erfolgreichem Commit:
    EventBus.publish(events)                -- Dispatch an Subscriber
```

Phase 2 startet mit **In-Process Dispatch** (ein Python-Prozess, Subscriber sind einfache Funktionsaufrufe nach Commit) — kein Message-Broker nötig bei aktuellem Volumen und einem einzigen Backend-Prozess. Der `EventBus` ist aber ein **Interface**, keine konkrete Implementierung im Domain Layer verankert:

```
EventBus (Interface)
  publish(events: list[Event]) -> None
  subscribe(event_type: str, handler: Callable) -> None
```

**Migrationspfad ohne Domain-Änderung:** Sobald mehrere Prozesse/Services nötig werden (z.B. Mobile-Realtime-Gateway als eigener Prozess), wird die In-Process-Implementierung durch eine Postgres-`LISTEN/NOTIFY`- oder Redis-Streams-Implementierung ersetzt — Domain Layer und Aggregates bleiben unverändert, weil sie nur gegen das `EventBus`-Interface programmieren. Das ist die konkrete Umsetzung von "neue Module hinzufügen, ohne bestehende anzupassen".

### Event Subscription

Zwei Subscriber-Kategorien, beide registrieren sich nur über `event_type`, ohne dass Aggregates sie kennen:

1. **Projection Builder** (Journal, Portfolio, PerformanceSnapshot) — schreiben in `projections.*`.
2. **Reactive Services** (Risk Monitor, WeeklyReport-Datensammlung) — können Folge-Commands auslösen.

Ein neues Modul (z.B. später "Tax Report Generator") registriert sich einfach als weiterer Subscriber auf `AllocationClosed` — kein bestehender Code wird angefasst.

### Projection Updates

Jede Projection führt einen Checkpoint in `projections.checkpoints` (`last_processed_seq`). Update-Loop:

```
seq = checkpoint.last_processed_seq
events = SELECT * FROM core.event_store WHERE global_seq > seq ORDER BY global_seq
for event in events:
    projection.apply(event)   -- muss idempotent sein
    checkpoint.last_processed_seq = event.global_seq
```

Idempotenz ist Pflicht (nicht optional), weil bei Prozess-Neustart derselbe Batch erneut verarbeitet werden könnte, falls der Checkpoint nicht exakt nach jedem Event persistiert wird (Batch-Commit alle N Events als Performance-Kompromiss ist erlaubt, solange `apply()` bei Wiederholung dasselbe Ergebnis liefert — z.B. durch `UPSERT` statt `INSERT`).

### Event Versioning

- `schema_version` im Envelope (2.1). Additive Payload-Änderungen (neues optionales Feld) erhöhen `schema_version` nicht zwingend. Breaking Changes (Feld umbenannt/Bedeutung geändert) verlangen entweder einen **Upcaster** (Funktion, die alte Payload-Form in neue transformiert, beim Lesen angewendet) oder — bei fachlicher Bedeutungsänderung — einen **neuen `event_type`** (z.B. `AllocationClosedV2`) statt stille Mutation der Bedeutung.
- Historische Events werden **nie** nachträglich umgeschrieben. Upcaster laufen nur im Lesepfad (Replay/Projection-Build), nie als Migration gegen `core.event_store`.

### Replay-Fähigkeit

Zwei Replay-Arten:
1. **Projection-Rebuild:** `TRUNCATE projections.<table>`, Checkpoint auf 0, Update-Loop von vorn — nötig nach Projection-Logik-Änderungen oder Korruption.
2. **Aggregate-State-Rebuild:** Für ein einzelnes Aggregate den Event-Stream (`aggregate_type, aggregate_id` gefiltert, nach `version` sortiert) neu abspielen, State-Tabellenzeile neu berechnen — nötig für Debugging/Audit oder falls die State-Tabelle je inkonsistent zum Event Store würde (sollte durch die atomare Transaktion in 2.1 nie vorkommen, aber die Fähigkeit ist der Sicherheitsnetz-Beweis, dass Event Store wirklich die alleinige Wahrheit ist).

---

## 2.4 Command Handling — vollständige Pipeline

```
1. CLIENT
   Desktop oder Mobile sendet Command (JSON, inkl. command_id [Client-generiert, für Idempotenz],
   correlation_id, issued_by [user_id + device_id])
        ↓
2. API LAYER — Authentifizierung + Autorisierung (2.7)
   - Session/Token gültig?
   - Gehört das Ziel-Aggregate (z.B. account_id im Payload) diesem User?
        ↓
3. IDEMPOTENZ-CHECK
   SELECT FROM core.processed_commands WHERE command_id = ?
   -- falls vorhanden: vorheriges Ergebnis zurückgeben, keine Doppelverarbeitung
   -- (kritisch für Mobile mit instabiler Verbindung: Retry darf nie doppelt schließen/öffnen)
        ↓
4. STRUKTURELLE VALIDIERUNG
   Pflichtfelder, Typen, Wertebereiche — kein Domain-Wissen, nur Command-Schema
        ↓
5. AGGREGATE LADEN
   Repository.load(aggregate_id) -> Aggregate (+ expected_version)
        ↓
6. RISK GATE (nur für kapitalrelevante Commands: ConfirmAllocationCommand,
   MarkAllocationOpenedCommand — in v1 beide manuell von Desktop/Mobile gesendet,
   s. Entscheidung 1: kein Broker-Webhook, User trägt Fill nach manueller Eröffnung
   beim Broker selbst ein)
   RiskGateService.evaluate(command, aggregate_state) -> ALLOW | ALLOW_WITH_ADJUSTMENT | REJECT
   -- bei REJECT: Pipeline bricht hier ab, Response = Ablehnungsgrund, KEIN Event
        ↓
7. DOMAIN-AUSFÜHRUNG
   aggregate.handle(command, risk_decision) -> Event(s) im Speicher, Invarianten geprüft
   -- Invariantenverletzung -> Exception, Pipeline bricht ab, KEIN Event
        ↓
8. PERSISTIEREN (UnitOfWork, eine Transaktion)
   BEGIN
     INSERT event_store (mit expected_version -> Optimistic-Concurrency-Check)
     UPDATE aggregate_state_table
     INSERT processed_commands (command_id, result_summary)
   COMMIT
        ↓
9. PUBLIZIEREN
   EventBus.publish(events) — nach erfolgreichem Commit
        ↓
10. PROJECTION UPDATES (asynchron, eventually consistent)
    Journal/Portfolio/Performance-Builder konsumieren, Risk Monitor konsumiert
        ↓
11. RESPONSE AN CLIENT
    Synchron: Ergebnis-Event(s) + finaler Aggregate-Status, oder Ablehnungsgrund (Schritt 4/6/7)
```

**Scope-Auflösung vor Schritt 6 (Entscheidung 2):** Der `PolicyScopeResolver` sammelt für den aktuellen Account (plus dessen Empire, dessen Prop-Firm-Template, den User, plus ggf. betroffenes Signal/Allocation) alle `ACTIVE`-Policies über die Scope-Hierarchie `GLOBAL < USER < EMPIRE < PROP_FIRM < ACCOUNT < SIGNAL < ALLOCATION`. Existieren zwei Policies mit demselben `policy_key` auf unterschiedlichen Ebenen (z.B. eine `max-drawdown`-Regel im PROP_FIRM-Template UND eine speziellere auf ACCOUNT-Ebene), **gewinnt die spezifischere** — die PROP_FIRM-Template-Regel wird für diesen Account vollständig durch die ACCOUNT-Regel ersetzt, nicht kombiniert. Policies mit unterschiedlichen `policy_key`s addieren sich (alle werden ausgewertet, `priority` entscheidet bei mehreren gleichzeitig ausgelösten die Reihenfolge der Aktionen, wie in `DOMAIN_ARCHITECTURE.md` §3.2 beschrieben).

**Zwei-Gate-Realität konkret in der Pipeline:** `ConfirmAllocationCommand` durchläuft Schritt 6 einmal (geplantes Risiko). Der spätere, separate `MarkAllocationOpenedCommand` durchläuft die **komplette Pipeline erneut von Schritt 2 an** — mit dem echten Fill-Preis im Payload — und damit Schritt 6 ein zweites Mal, unabhängig vom Ergebnis des ersten Durchlaufs. Es gibt keinen Zustand, der den zweiten Gate-Durchlauf umgehen kann.

**Command-Idempotenz-Tabelle:**
```sql
CREATE TABLE core.processed_commands (
    command_id      UUID PRIMARY KEY,
    aggregate_id    UUID NOT NULL,
    result_summary  JSONB NOT NULL,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 2.5 API Layer — Struktur

**Grundsatzentscheidung: Typisierte Endpunkte pro Command/Query, kein generischer RPC-Bucket.**

*Verglichen:* Ein einziger `/commands`-Endpunkt mit `command_type` im Body wäre weniger Boilerplate, aber verliert OpenAPI-Typsicherheit pro Command, erschwert Versionierung einzelner Commands und macht die API für einen zweiten Entwickler (oder dich in 3 Jahren) schwerer entdeckbar. Für ein System, das 5-10 Jahre lesbar bleiben soll, überwiegt Explizitheit.

**Drei Oberflächenarten:**

| Bereich | Commands | Queries | Realtime | Begründung |
|---|---|---|---|---|
| **Account & Empire** | wenige (manuelle Bestätigungen) | ja (Balance/Equity) | **ja, hohe Priorität** | Drawdown-Änderungen sind sicherheitskritisch — müssen sofort beim User ankommen, nicht erst beim nächsten Poll |
| **Execution (Allocation)** | Confirm, Cancel, MarkOpened | ja (offene Positionen) | **ja** | Status-Änderungen (Fill, Close) müssen auf Mobile sofort sichtbar sein |
| **Risk** | keine direkten Client-Commands (Gate läuft intern) | ja (Entscheidungs-Log: warum wurde X abgelehnt) | **ja** | `RiskPolicyTriggered` (z.B. Daily-DD erreicht) muss als Alert gepusht werden |
| **Journal** | AddJournalNoteCommand, EditJournalNoteCommand | ja (Journal-Timeline) | nein | nicht sicherheitskritisch, Laden beim App-Öffnen reicht |
| **Research** | GenerateSignalsCommand (nur Desktop) | ja (aktuelle Signale) | optional | Signal-Generierung ist ohnehin ein Desktop-Batch-Vorgang |
| **Calendar** | keine Client-Commands (nur Scheduler-intern) | ja | nein | reine Faktenquelle |
| **Portfolio/Performance** | keine | ja | nein | reine Projektion, periodisches Neuladen ausreichend |

**Realtime-Mechanismus:** WebSocket- oder SSE-Kanal pro authentifizierter Session, gespeist durch einen `RealtimeGateway`-Subscriber, der ausgewählte Event-Typen an verbundene Clients weiterleitet — gefiltert auf den `user_id`-Scope des Clients (kein Cross-User-Leak, wichtig sobald Mehrbenutzer aktiv wird). Technisch ist das nur ein weiterer Event-Bus-Subscriber (2.3), kein Sonderpfad.

**Authentication:** jede Anfrage (Command wie Query) läuft durch denselben Auth-Layer (2.7) — keine Sonderregel für Queries.

---

## 2.6 Synchronisation

```
Desktop  →  Cloud (PostgreSQL, einzige Wahrheit)  →  Mobile
```

**Desktop:** Bleibt alleiniger Ort für Heavy Computing (ML-Modelle, Backtests, Report-Generierung — unverändert gegenüber Vision). Nur die **Ergebnisse** (ein generiertes Signal, ein veröffentlichter Report) werden über Commands an das Backend übergeben und damit Teil der Single Source of Truth. Desktop selbst hält keine eigene dauerhafte Kopie des Zustands — es liest Projektionen genau wie Mobile.

**Cloud:** Ein PostgreSQL-Cluster (verwaltet, z.B. Supabase/RDS) — die einzige Instanz von `core` + `projections`. Kein Client besitzt eine eigene "Wahrheit".

**Mobile:** Sendet die in 2.5 gelisteten Commands, liest Projektionen, empfängt Realtime-Events.

**Offline Recovery:**
- **Client-seitige Outbox:** Commands werden bei fehlender Verbindung lokal in einer Warteschlange (mit `command_id`) gehalten und beim Reconnect nacheinander gesendet. Server-seitige Idempotenz (2.4, `processed_commands`) verhindert Doppelverarbeitung bei Retry.
- **Lesbarkeit im Offline-Zustand:** letzter bekannter Projektionsstand wird client-seitig gecacht, klar als "Stand: [Zeitstempel]" markiert — kein stiller Vortäusch von Aktualität.

**Konfliktlösung:**
Da jedes Aggregate serverseitig Optimistic Concurrency erzwingt (`expected_version`), führt ein verspäteter Offline-Command, dessen zugrunde liegender Zustand inzwischen überholt ist, zu einer klaren, expliziten Ablehnung ("Zustand veraltet, bitte aktualisieren") statt zu stillem Überschreiben. Das gilt **symmetrisch für Desktop und Mobile** — kein Client hat Vorrang, was direkt die Prinzip-Vorgabe "gleichberechtigte Clients" erfüllt. Bei aktuell nur einem User sind echte Konflikte selten, aber der Mechanismus ist bereits mehrbenutzerfähig (kein Nacharbeiten nötig, wenn später ein zweites Gerät/eine zweite Person hinzukommt).

---

## 2.7 Security

**Grundsatz:** Ein User heute, Architektur für viele User ab Tag 1 — additive Ownership-Attribute, keine spätere Umstrukturierung.

```sql
CREATE TABLE core.users (
    id             UUID PRIMARY KEY,
    email          TEXT NOT NULL UNIQUE,
    password_hash  TEXT NULL,          -- oder external IdP subject, je nach Auth-Provider-Wahl
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE core.devices (
    id             UUID PRIMARY KEY,
    user_id        UUID NOT NULL REFERENCES core.users(id),
    device_type    TEXT NOT NULL CHECK (device_type IN ('DESKTOP','MOBILE','SYSTEM')),
    device_name    TEXT NOT NULL,
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at   TIMESTAMPTZ NULL,
    revoked_at     TIMESTAMPTZ NULL
);

CREATE TABLE core.sessions (
    id             UUID PRIMARY KEY,
    user_id        UUID NOT NULL REFERENCES core.users(id),
    device_id      UUID NOT NULL REFERENCES core.devices(id),
    issued_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at     TIMESTAMPTZ NOT NULL,
    revoked_at     TIMESTAMPTZ NULL
);

CREATE TABLE core.api_keys (        -- für Scheduled Jobs / System-Commands (SyncCalendarCommand etc.)
    id             UUID PRIMARY KEY,
    user_id        UUID NOT NULL REFERENCES core.users(id),
    purpose        TEXT NOT NULL,     -- z.B. 'calendar-scraper'
    scopes         TEXT[] NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at     TIMESTAMPTZ NULL
);

CREATE TABLE core.roles (
    id    TEXT PRIMARY KEY,           -- 'OWNER', künftig z.B. 'VIEWER'
    name  TEXT NOT NULL
);

CREATE TABLE core.user_roles (
    user_id  UUID NOT NULL REFERENCES core.users(id),
    role_id  TEXT NOT NULL REFERENCES core.roles(id),
    PRIMARY KEY (user_id, role_id)
);
```

**Auth-Fluss:** JWT-Access-Token (kurzlebig) + Refresh-Token (langlebig, an `device_id` gebunden) — Standardmuster, keine Eigenentwicklung nötig (z.B. via Supabase Auth oder eigenständigem Auth-Service).

**Autorisierung (in API Layer, vor Schritt 3 der Command-Pipeline):** Jeder Command/Query-Payload referenziert ein Ziel-Aggregate (meist über `account_id`/`empire_id`). Die Prüfung "gehört `account_id` dem `user_id` der Session" läuft als Join über `core.accounts.user_id` — das ist der Grund, warum `user_id` additiv auf `accounts`/`empires` ergänzt wird (Konflikt 3 aus Abschnitt 0).

**Rollen schon jetzt, auch mit 1 User:** `OWNER`-Rolle wird angelegt und dem einzigen User zugewiesen. Erweiterung um z.B. `VIEWER` (read-only Zugriff auf Portfolio für einen Mentor/Steuerberater) ist dann nur ein neuer Rolleneintrag + Permission-Check in der Query-API, keine Schema-Änderung.

**Jedes Event trägt `device_id`** (2.1) — vollständiger Audit-Trail: "welches Gerät hat wann welchen Command ausgelöst", unabhängig von User-Mehrzahl bereits heute nützlich für Debugging.

---

## 2.8 Testing Strategy

| Ebene | Was wird getestet | Beispiel |
|---|---|---|
| **Unit Tests** | Aggregate-Invarianten isoliert, ohne DB | `TradeAllocation` kann nicht `OPEN` ohne vorheriges `CONFIRMED` + gültige Risk-Decision |
| **Domain Tests** | Policy-Entscheidungslogik, tabellengetrieben je Policy | Same-Pair-Policy: gegebene offene Allocation im selben Pair → `ALLOW_WITH_ADJUSTMENT(0.5pp)` |
| **Event Tests** | Envelope-Contract-Konformität, Versionsinkrement, Replay-Determinismus | Jedes erzeugte Event hat `aggregate_type/id/version/timestamp/source`; zweimaliges Replay eines Streams liefert identischen State |
| **Integration Tests** | Vollständige Command-Pipeline gegen echte Test-Postgres | `ConfirmAllocationCommand` → Gate → Event → State-Tabelle aktualisiert, in einer Transaktion |
| **Replay Tests** | Projection-Rebuild vs. vorheriger Live-Stand | `journal_view` nach Rebuild identisch zu vorherigem Stand (Regression-Schutz bei Projection-Änderungen) |
| **End-to-End Tests** | Realistische Multi-Step-Flows über die API | Propose→Confirm[Gate1]→MarkOpened[Gate2]→PartialHit→Close→Journal-Note→Performance-Snapshot→Portfolio-Update |

**Zwingend automatisiert zu testende kritische Prozesse (nicht verhandelbar):**

1. Kein `AllocationOpened` ohne frisches `ALLOW` aus **Gate-Durchlauf 2** — ein veralteter/erster Durchlauf darf niemals ausreichen.
2. Same-Pair/Correlation/Blackout-Ablehnungen bzw. -Anpassungen erzeugen korrektes `applied_risk_pct` im Event-Payload.
3. `accounts.balance` ändert sich **ausschließlich** über die `AllocationClosed`-Reaktion — DB-Rollenrechte + Integrationstest, der einen direkten Schreibversuch von außen verifiziert ablehnt.
4. `EmpireStageAdvanced` entsteht **nie** ohne vorheriges `ConfirmEmpireStageAdvanceCommand` — kein Codepfad darf `EmpireStageAdvanceProposed` überspringen.
5. `journal_notes` enthält strukturell keine Trade-Fakten (Schema-Test: Spaltenliste enthält keine Ergebnis-/Preis-/Status-Felder).
6. Optimistic-Concurrency-Race: zwei gleichzeitige Commands auf demselben Aggregate → genau einer gewinnt, der andere erhält einen sauberen Konfliktfehler, kein Lost Update.
7. Vollständiger Replay des Event Stores reproduziert alle Aggregate-State-Tabellen und Projektionen deterministisch (bis auf berechnete Zeitstempel).

---

## 2.9 Technische Roadmap

Reihenfolge minimiert Refactoring, indem jede Stufe auf einem bereits stabilen Fundament der vorherigen aufbaut — abgeleitet aus der fachlichen Roadmap in `DOMAIN_ARCHITECTURE.md` §7, um eine Stufe (0) ergänzt, die dort bewusst nicht Teil der fachlichen Betrachtung war.

**Stufe 0 — Technisches Fundament (neu, vor allem anderen)**
`core.event_store`, Shared Kernel (`AggregateRoot`, `EventEnvelope`, `UnitOfWork`, `EventBus`-Interface, `Repository`-Interface), `processed_commands`-Dedup-Tabelle, minimales `users`/`devices`/`sessions`-Schema. Ohne dieses Fundament kann kein Aggregate korrekt gebaut werden — bewusst vor Stufe 1 der fachlichen Roadmap eingeschoben.

**Stufe 1 — Account + Empire**
Aggregates, Repositories, State-Tabellen. Noch keine API. Verifiziert über Unit-/Integrationstests. Bestehendes Empire-Modul (64 Tests) dient als fachliche Testreferenz, wird aber nicht 1:1 übernommen, da es nicht event-sourced war.

**Stufe 2 — TradeAllocation + Risk Gate (SYNC_GATE) + Risk Monitor (ASYNC_MONITOR)**
Gemeinsam, wie fachlich gefordert. Hier läuft die vollständige Command-Pipeline (2.4) erstmals komplett durch, inkl. beider Gate-Durchläufe.

**Stufe 3 — Command API (nur Stufe-1/2-Endpunkte) + Auth-Grundgerüst**
Ab hier spricht Desktop erstmals real gegen das Backend statt gegen lokale Dateien/DataFrames.

**Stufe 4 — JournalNote-Aggregate + Journal-Projection + Realtime-Grundgerüst**
Erste reine Projection (kein Risk-Bezug) — etabliert das Subscriber-Pattern für zukünftige Module.

**Stufe 5 — Research (Signal, Playbook, Calendar)**
Dockt an stabiles Event-Fundament an, ohne Execution zu berühren.

**Stufe 6 — Portfolio & PerformanceSnapshot Projections + zugehörige Query-API**

**Stufe 7 — WeeklyReport-Integration**

**Stufe 8 — Mobile-taugliche API-Vervollständigung**
Realtime für Allocation-Status, vollständige Geräte-Registrierung, serverseitiger Vertrag für die Offline-Command-Queue (2.6) final abgesichert — bewusst zuletzt, analog zur fachlichen Entscheidung, Mobile nicht gegen ein bewegliches Ziel zu bauen.

---

## Phase 2 — EINGEFROREN

Alle drei offenen Punkte sind entschieden (Abschnitt 0, Entscheidungen 1–3) und im Schema/Domain-Layer eingearbeitet:

1. `MarkAllocationOpenedCommand` wird in v1 manuell von Desktop/Mobile gesendet — Broker-Integration ist eine spätere Adapter-Erweiterung ohne Architekturänderung.
2. RiskPolicy-Scope ist eine Ebenen-Hierarchie (`GLOBAL/USER/EMPIRE/PROP_FIRM/ACCOUNT/SIGNAL/ALLOCATION`) mit Override nach Spezifität über `policy_key`, aufgelöst vom `PolicyScopeResolver`.
3. `user_id` ist ab Tag 1 auf `accounts`, `empires`, `signals`, `weekly_reports` vorhanden; `trade_allocations` und `playbooks` bleiben bewusst ohne eigene `user_id` (Ownership transitiv bzw. geteiltes Referenzwissen). Keine Team-/Org-Logik in v1.

Damit ist Phase 2 vollständig eingefroren.

---

## Implementierungsreihenfolge (bestätigt)

Strikte Reihenfolge für die Implementierungsphase, wie von dir vorgegeben:

1. **Event Store + Shared Kernel** (`core.event_store`, `AggregateRoot`, `EventEnvelope`, `UnitOfWork`, `EventBus`-Interface, `Repository`-Interface, `processed_commands`)
2. **Account + Empire State** (Aggregates, Repositories, State-Tabellen, inkl. `users`/`devices` minimal für `user_id`-FKs)
3. **Allocation Lifecycle** (TradeAllocation-Aggregate, State-Machine, `allocation_lifecycle_service`)
4. **Risk Gate** (RiskPolicy-Aggregate, `PolicyScopeResolver`, `risk_gate_service`, erste konkrete Policies: Same Pair, Max Drawdown)
5. **Journal Projection** (JournalNote-Aggregate + Projection Builder)
6. **API** (Command-/Query-Endpunkte für die bis dahin gebauten Contexts, Auth-Grundgerüst)
7. **Mobile MVP** (Realtime-Grundgerüst + minimale Mobile-taugliche Endpunkte)

Das deckt sich mit der technischen Roadmap in Abschnitt 2.9 (Stufe 0–4 + Teile von 8), komprimiert auf die Reihenfolge, die du für die Implementierung priorisierst — Research/Calendar/WeeklyReport/Portfolio (Stufen 5–7) folgen danach.
