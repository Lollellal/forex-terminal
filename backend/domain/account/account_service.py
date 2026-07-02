"""Reine Koordination: Command entgegennehmen, Aggregate bauen/laden,
speichern. Keine Invarianten hier — die liegen im Aggregate (account.py).
Existiert primär, damit der API Layer (Schritt 6) gegen einen Service statt
direkt gegen Repository + Aggregat-Konstruktoren programmiert, konsistent
mit AllocationLifecycleService/JournalNoteService."""

from __future__ import annotations

import uuid

from sqlalchemy import Connection

from .account import Account
from .account_repository import AccountRepository
from .commands import CreateAccountCommand, UpdateAccountBalanceCommand


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
