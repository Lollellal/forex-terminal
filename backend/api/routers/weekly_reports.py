"""WeeklyReport-Endpunkte. Registriert vom Desktop bereits generierte PDFs
in Supabase Storage + core.weekly_reports — keine Report-Generierung, keine
KI-Logik hier. Siehe MOBILE_DATA_CONTRACT.md (Schritt 7, Abschnitt "Weekly
Report") für den ursprünglich spezifizierten Ziel-Contract."""

from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import Connection, text

from backend.api.dependencies import get_read_conn, get_write_conn
from backend.api.schemas.weekly_report import WeeklyReportDownloadUrlResponse, WeeklyReportResponse
from backend.domain.weekly_report.commands import PublishWeeklyReportCommand, RegisterWeeklyReportCommand
from backend.domain.weekly_report.weekly_report_repository import WeeklyReportRepository
from backend.domain.weekly_report.weekly_report_service import WeeklyReportService
from backend.infrastructure.event_store import EventStore
from backend.infrastructure.projections import ProjectionRunner
from backend.infrastructure.storage import WeeklyReportStorage

router = APIRouter(prefix="/weekly-reports", tags=["weekly-reports"])

_storage = WeeklyReportStorage()
_service = WeeklyReportService(WeeklyReportRepository(EventStore()), _storage)
_projections = ProjectionRunner(EventStore())


@router.post("", response_model=WeeklyReportResponse, status_code=201)
async def register_weekly_report(
    user_id: uuid.UUID = Form(...),
    period_start: date = Form(...),
    period_end: date = Form(...),
    file: UploadFile = File(...),
    conn: Connection = Depends(get_write_conn),
):
    content = await file.read()
    report = _service.register(
        conn,
        RegisterWeeklyReportCommand(
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            content=content,
            content_type=file.content_type or "application/pdf",
        ),
    )
    _projections.catch_up(conn)
    return _to_response(report)


@router.post("/{report_id}/publish", response_model=WeeklyReportResponse)
def publish_weekly_report(report_id: uuid.UUID, conn: Connection = Depends(get_write_conn)):
    report = _service.publish(conn, PublishWeeklyReportCommand(report_id=report_id))
    _projections.catch_up(conn)
    return _to_response(report)


@router.get("/{report_id}/download-url", response_model=WeeklyReportDownloadUrlResponse)
def get_download_url(report_id: uuid.UUID, conn: Connection = Depends(get_read_conn)):
    row = conn.execute(
        text("SELECT content_ref FROM projections.weekly_reports WHERE id = :id"), {"id": str(report_id)}
    ).first()
    if row is None or row[0] is None:
        raise HTTPException(status_code=404, detail=f"WeeklyReport {report_id} nicht gefunden")
    url = _storage.signed_url(row[0])
    return WeeklyReportDownloadUrlResponse(url=url, expires_in=3600)


def _to_response(report) -> WeeklyReportResponse:
    return WeeklyReportResponse(
        id=report.id,
        user_id=report.user_id,
        period_start=report.period_start,
        period_end=report.period_end,
        status=report.status,
        content_ref=report.content_ref,
        published_at=report.published_at,
    )
