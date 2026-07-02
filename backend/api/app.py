"""FastAPI-App-Factory. Typisierte Command-/Query-Endpunkte auf Basis der
bestehenden Domain-Services (Implementierungsschritt 6) — keine neue
Business-Logik, jeder Endpunkt ist eine dünne Übersetzung Request/Response
<-> Command/Query gegen die bereits getesteten Aggregate aus den Schritten
2-5. Desktop und Mobile sollen künftig dieselbe API nutzen
(BACKEND_ARCHITECTURE.md §2.5/§2.6).

Bewusst ohne Auth: core.users/sessions/devices (§2.7) sind noch nicht
gebaut (nur der minimale users-Stub aus Schritt 2) — diese API ist damit
für den Moment nur für vertrauenswürdige Netzwerke/lokale Nutzung gedacht,
kein öffentlicher Deploy. Auth kommt als eigener Schritt.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routers import (
    accounts,
    allocations,
    empires,
    journal,
    market_snapshot,
    portfolio,
    weekly_reports,
)
from backend.domain.risk.exceptions import RiskGateRejectedError
from backend.domain.shared.exceptions import ConcurrencyConflictError, EventStreamGapError


def create_app() -> FastAPI:
    app = FastAPI(title="Trading OS API", version="0.8.0")

    # Browser-Clients (Mobile-Web-App, Schritt 9) laufen auf einem anderen
    # Origin als die API — ohne CORS blockt der Browser jeden Request.
    # Offen für alle Origins, weil es (noch) keine Auth gibt (siehe oben) —
    # sobald Auth kommt, wird das auf konkrete Origins eingeschränkt.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(accounts.router)
    app.include_router(empires.router)
    app.include_router(allocations.router)
    app.include_router(journal.router)
    app.include_router(portfolio.router)
    app.include_router(weekly_reports.router)
    app.include_router(market_snapshot.router)

    @app.exception_handler(LookupError)
    def handle_not_found(request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(RiskGateRejectedError)
    def handle_risk_gate_rejected(request: Request, exc: RiskGateRejectedError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": exc.decision.reason, "triggered_policy": exc.decision.triggered_policy},
        )

    @app.exception_handler(ConcurrencyConflictError)
    def handle_concurrency_conflict(request: Request, exc: ConcurrencyConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(EventStreamGapError)
    def handle_event_stream_gap(request: Request, exc: EventStreamGapError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


app = create_app()
