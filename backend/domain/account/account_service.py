"""Reine Koordination: Command entgegennehmen, Aggregate bauen/laden,
speichern. Keine Invarianten hier — die liegen im Aggregate (account.py).
Existiert primär, damit der API Layer (Schritt 6) gegen einen Service statt
direkt gegen Repository + Aggregat-Konstruktoren programmiert, konsistent
mit AllocationLifecycleService/JournalNoteService."""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy import Connection, text

from .account import Account
from .account_repository import AccountRepository
from .commands import CreateAccountCommand, UpdateAccountBalanceCommand

_CLOSED_ALLOCATIONS_SQL = text(
    "SELECT applied_risk_pct, realized_r FROM core.trade_allocations "
    "WHERE account_id = :account_id AND status = 'CLOSED'"
)


class AccountService:
    def __init__(self, repository: AccountRepository) -> None:
        self._repository = repository

    def create(self, conn: Connection, command: CreateAccountCommand) -> Account:
        account = Account.create(
            uuid.uuid4(),
            user_id=command.user_id,
            empire_id=command.empire_id,
            account_type=command.account_type,
            initial_balance=command.initial_balance,
            initial_equity=command.initial_equity,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, account)
        return account

    def update_balance(self, conn: Connection, command: UpdateAccountBalanceCommand) -> Account:
        account = self._repository.load(conn, command.account_id)
        account.update_balance(
            balance=command.balance,
            equity=command.equity,
            source=command.source,
            correlation_id=command.correlation_id,
        )
        self._repository.save(conn, account)
        return account

    def recompute_from_closed_allocations(self, conn: Connection, account_id: uuid.UUID) -> Account:
        """Fuehrt Balance/Equity aus allen CLOSED-Allocations des Accounts
        nach. Fixed-Fractional gegen initial_balance (kein Compounding) --
        identische Formel wie compute_account_stats in src/empire.py, damit
        Zahlen zwischen Legacy-Anzeige und Backend konsistent bleiben. Wird
        von der Allocation-Close-Koordination aufgerufen (siehe
        AllocationLifecycleService), nicht Teil des Allocation-Aggregates
        selbst (Trennung Account/Allocation, wie in BACKEND_ARCHITECTURE.md
        beschrieben)."""
        account = self._repository.load(conn, account_id)
        rows = conn.execute(_CLOSED_ALLOCATIONS_SQL, {"account_id": str(account_id)}).all()
        pnl = sum(
            (account.initial_balance * Decimal(row.applied_risk_pct) / Decimal(100) * Decimal(row.realized_r))
            for row in rows
        ) if rows else Decimal("0")
        new_balance = account.initial_balance + pnl
        return self.update_balance(
            conn,
            UpdateAccountBalanceCommand(account_id=account_id, balance=new_balance, equity=new_balance),
        )
