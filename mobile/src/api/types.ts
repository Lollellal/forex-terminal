// Spiegelt exakt die Pydantic-Response-Schemas aus backend/api/schemas/*.py.
// Decimal-Felder serialisieren als JSON-Strings (siehe test_api.py) -- nie
// als number behandeln, außer nach expliziter Parse fürs Rendern.

export type AllocationStatus = "CREATED" | "CONFIRMED" | "OPEN" | "CLOSED";
export type Direction = "LONG" | "SHORT";
export type AccountType = "LIVE" | "PROP_FIRM";
export type CloseReason = "SL" | "TP" | "TIME_EXIT" | "FORCE_CLOSE" | "MANUAL";

export interface EmpireSummary {
  empire_id: string;
  name: string;
  account_count: number;
  total_balance: string;
  total_equity: string;
}

export interface StandaloneAccountSummary {
  account_id: string;
  account_type: AccountType;
  status: string;
  balance: string;
  equity: string;
}

export interface RecentJournalEntry {
  allocation_id: string;
  pair: string;
  status: AllocationStatus;
  updated_at: string;
}

export interface PortfolioResponse {
  user_id: string;
  total_balance: string;
  total_equity: string;
  empires: EmpireSummary[];
  standalone_accounts: StandaloneAccountSummary[];
  active_trade_count: number;
  recent_journal_entries: RecentJournalEntry[];
}

// Unveränderlicher Signal-Snapshot, eingefroren beim Desktop-Cloud-Publish
// (siehe src/trade_snapshot.py). Felder ohne echte Quelle sind null, nicht
// erfunden -- so rendern, dass null ehrlich als "nicht verfügbar" erscheint.
export interface SignalSnapshot {
  ml_score: number | null;
  quality: string | null;
  edge: string | null;
  alignment: string | null;
  combo_key: string | null;
  regime: string | null;
  overall_score: number | null;
  seasonality_score: number | null;
  primary_drivers: string[] | null;
  weekly_report_id: string | null;
  report_week: string | null;
  // Timeline + Makro-Begruendung, siehe src/trade_snapshot.py::build_trade_context()
  be_date: string | null;
  close_date: string | null;
  be_trading_days: number | null;
  max_hold_trading_days: number | null;
  why_base_ccy: string | null;
  why_base_section: string | null;
  why_quote_ccy: string | null;
  why_quote_section: string | null;
}

export interface AllocationOverviewResponse {
  allocation_id: string;
  account_id: string;
  pair: string;
  direction: Direction;
  status: AllocationStatus;
  planned_risk_pct: string;
  applied_risk_pct: string | null;
  opened_at: string | null;
  closed_at: string | null;
  close_reason: CloseReason | null;
  realized_r: string | null;
  signal_snapshot: SignalSnapshot | null;
}

export interface JournalNoteEntry {
  note_id: string;
  text: string;
  attachments: string[];
  created_at: string;
  edited_at?: string;
}

export interface JournalViewResponse {
  allocation_id: string;
  account_id: string;
  pair: string;
  direction: Direction;
  status: AllocationStatus;
  planned_risk_pct: string;
  applied_risk_pct: string | null;
  closed_at: string | null;
  close_reason: CloseReason | null;
  realized_r: string | null;
  account_snapshot: { account_type: AccountType; balance: string; equity: string };
  notes: JournalNoteEntry[];
}

export interface EmpireOverviewResponse {
  empire_id: string;
  name: string;
  account_count: number;
  total_balance: string;
  total_equity: string;
}

export interface AccountBalanceProjectionResponse {
  account_id: string;
  empire_id: string | null;
  account_type: AccountType;
  status: string;
  balance: string;
  equity: string;
}

export interface WeeklyReportResponse {
  id: string;
  user_id: string;
  period_start: string;
  period_end: string;
  status: "DRAFT" | "GENERATED" | "PUBLISHED" | "ARCHIVED";
  content_ref: string | null;
  summary: string | null;
  published_at: string | null;
}

export interface WeeklyReportDownloadUrlResponse {
  url: string;
  expires_in: number;
}

export interface MarketSnapshotResponse {
  regime: string;
  vix: string | null;
  yield_curve: string | null;
  updated_at: string;
}
