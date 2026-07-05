"""Minimale Bearer-Token-Auth für die API — ein einziger Nutzer, kein
User-/Session-System. Das Secret liegt als API_AUTH_TOKEN in .env (bzw.
später in den Env-Vars des Hosting-Anbieters) und wird global gegen den
Authorization-Header jedes Requests geprüft (siehe app.py)."""

from __future__ import annotations

import os
import secrets

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()


def get_api_auth_token() -> str:
    token = os.environ.get("API_AUTH_TOKEN")
    if not token:
        raise RuntimeError(
            "API_AUTH_TOKEN ist nicht gesetzt. Trage ein zufälliges Secret als "
            "API_AUTH_TOKEN in .env ein (z.B. `python -c \"import secrets; "
            'print(secrets.token_urlsafe(32))"`).'
        )
    return token


def require_auth_token(authorization: str | None = Header(default=None)) -> None:
    expected = get_api_auth_token()
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Fehlender oder ungültiger Authorization-Header")
    provided = authorization.removeprefix("Bearer ")
    # Konstant-Zeit-Vergleich statt "==", um Timing-Angriffe auf das Secret zu vermeiden.
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Ungültiger Token")
