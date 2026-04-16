from fastapi import APIRouter, Depends, File, UploadFile, Header, HTTPException, status, Request, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, Dict
import datetime

from models.users import ResponseSchema, RecipientUpdate, CareRecipientCreate, ConditionInput, MedicationInput, AllergyInput
from tables.users import CareRecipient, CareTaker
from config import get_db
from repository.users import UsersRepo
from repository.medical_reports import create_medical_report, list_reports_for_recipient, delete_medical_report
from utils.summarizer import extract_text_from_bytes, summarize_text_via_gemini, summarize_report_insights_via_gemini
from fastapi.responses import StreamingResponse
from io import BytesIO
from tables.medical_reports import MedicalReport
from pydantic import BaseModel


router = APIRouter(tags=["Recipients"])


def _run_medical_history_pipeline(recipient_id: int, report_id: int, filename: str):
    """Async background task. Runs V2 hybrid pipeline; falls back to V1 on failure."""
    try:
        _run_medical_history_pipeline_v2(recipient_id, report_id, filename)
    except Exception:
        _run_medical_history_pipeline_v1_legacy(recipient_id, report_id, filename)


def _run_medical_history_pipeline_v2(recipient_id: int, report_id: int, filename: str):
    """V2 hybrid pipeline: deterministic lab extraction + Gemini for clinical context."""
    from config import SessionLocal
    db = SessionLocal()
    try:
        from tables.medical_reports import MedicalReport, ReportProcessingStatus
        from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity, SourceType
        from services.report_ingestion import run_hybrid_lab_extraction, extract_structured_report
        from services.disease_detection import detect_diseases_from_report
        from services.disease_progression import analyze_progression
        from services.alert_engine import generate_alerts, check_monitoring_gaps
        from services.medical_history_ai import calculate_risk_score
        from repository import medical_history as repo
        import datetime

        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if not report:
            return

        report.processing_status = ReportProcessingStatus.processing
        db.flush()

        # ── 1. Text extraction ────────────────────────────────────────────────
        text = ""
        if report.data:
            try:
                text = extract_text_from_bytes(report.data, filename)
            except Exception as e:
                print(f"[pipeline_v2] Text extraction failed: {e}")

        if not text:
            print(f"[pipeline_v2] No text extracted from report {report_id}")
            report.processing_status = ReportProcessingStatus.failed
            db.commit()
            return

        # ── 2. Hybrid lab extraction (rule → fuzzy → LLM fallback) ─────────
        upload_date = report.uploaded_at.date() if report.uploaded_at else datetime.date.today()
        hybrid_result = run_hybrid_lab_extraction(
            text=text, report_id=report_id,
            care_recipient_id=recipient_id, upload_date=upload_date, db=db,
        )
        report_date_obj = hybrid_result["report_date"]
        if report_date_obj:
            report.report_date = report_date_obj
        print(f"[pipeline_v2] Lab rows saved={hybrid_result['saved_count']} "
              f"template={hybrid_result['template']} date={report_date_obj}")

        # ── 3. Gemini: diagnoses, medications, notes (NOT lab values) ──────
        report_date_str = str(report_date_obj) if report_date_obj else None
        extracted = extract_structured_report(text, report_date_hint=report_date_str)
        extracted["lab_values"] = {
            row["metric_name"]: {"value": row["normalized_value"], "unit": row["normalized_unit"]}
            for row in hybrid_result["lab_rows"]
        }
        extracted["extraction_template"] = hybrid_result["template"]
        report.extracted_data = extracted

        # ── 4. Disease detection ──────────────────────────────────────────────
        existing = repo.get_all_conditions(db, recipient_id)
        new_diseases = detect_diseases_from_report(
            extracted, existing, db, report_id,
            report_date_str or extracted.get("report_date")
        )
        for disease in new_diseases:
            cond = PatientCondition(
                care_recipient_id=recipient_id,
                disease_code=disease["disease_code"],
                disease_name=disease["disease_name"],
                status=ConditionStatus.active,
                severity=ConditionSeverity.moderate,
                first_detected=disease.get("first_detected", datetime.date.today()),
                last_updated=disease.get("first_detected", datetime.date.today()),
                baseline_value=disease.get("baseline_value"),
                baseline_date=disease.get("baseline_date"),
                confidence_score=disease.get("confidence_score", 0.5),
                source_type=SourceType(disease.get("source_type", "lab_inferred")),
                source_report_id=report_id,
            )
            db.add(cond)
        db.flush()

        # ── 5–7. Progression, alerts, risk score ──────────────────────────────
        progression = analyze_progression(
            recipient_id, extracted, report_id,
            report_date_str or extracted.get("report_date", ""), db
        )
        generate_alerts(recipient_id, progression, db)
        check_monitoring_gaps(recipient_id, db)
        calculate_risk_score(recipient_id, db)

        # ── 8. Update last_report_date ────────────────────────────────────────
        from tables.users import CareRecipient
        rec = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
        if rec:
            rec.last_report_date = datetime.datetime.utcnow()

        report.processing_status = ReportProcessingStatus.completed
        db.commit()
        print(f"[pipeline_v2] ✅ Report {report_id} — "
              f"{hybrid_result['saved_count']} lab values, {len(new_diseases)} new conditions")

        # ── 9. Aggregate summary ──────────────────────────────────────────────
        try:
            _recalculate_aggregate_summary(db, recipient_id)
        except Exception as e:
            print(f"[pipeline_v2] summary failed: {e}")

        # ── 10. Clinical Recommendations ──────────────────────────────────────
        try:
            from services.recommendation_engine import generate_recommendations
            generate_recommendations(recipient_id, db)
        except Exception as e:
            print(f"[pipeline_v2] recommendations failed: {e}")

    except Exception as e:
        print(f"[pipeline_v2] ❌ Error: {e}")
        import traceback; traceback.print_exc()
        db.rollback()
        try:
            rep = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
            if rep:
                rep.processing_status = ReportProcessingStatus.failed
                db.commit()
        except Exception:
            pass
        raise   # Re-raise so the outer dispatcher triggers v1 fallback
    finally:
        db.close()


def _run_medical_history_pipeline_v1_legacy(recipient_id: int, report_id: int, filename: str):
    """V1 legacy fallback — pure Gemini extraction (used only if v2 fails entirely)."""
    from config import SessionLocal
    db = SessionLocal()
    try:
        from tables.medical_reports import MedicalReport, ReportProcessingStatus
        from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity, SourceType
        from services.report_ingestion import extract_structured_report
        from services.disease_detection import detect_diseases_from_report
        from services.disease_progression import analyze_progression
        from services.alert_engine import generate_alerts, check_monitoring_gaps
        from services.medical_history_ai import calculate_risk_score
        from repository import medical_history as repo
        import datetime

        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if not report:
            return

        text = extract_text_from_bytes(report.data, filename) if report.data else ""
        if not text:
            report.processing_status = ReportProcessingStatus.failed
            db.commit()
            return

        extracted = extract_structured_report(text)
        report.extracted_data = extracted
        if extracted.get("report_date"):
            try:
                report.report_date = datetime.date.fromisoformat(extracted["report_date"])
            except (ValueError, TypeError):
                pass

        existing = repo.get_all_conditions(db, recipient_id)
        new_diseases = detect_diseases_from_report(extracted, existing, db, report_id, extracted.get("report_date"))
        for disease in new_diseases:
            db.add(PatientCondition(
                care_recipient_id=recipient_id,
                disease_code=disease["disease_code"],
                disease_name=disease["disease_name"],
                status=ConditionStatus.active,
                severity=ConditionSeverity.moderate,
                first_detected=disease.get("first_detected", datetime.date.today()),
                last_updated=disease.get("first_detected", datetime.date.today()),
                confidence_score=disease.get("confidence_score", 0.5),
                source_type=SourceType(disease.get("source_type", "lab_inferred")),
                source_report_id=report_id,
            ))
        db.flush()
        progression = analyze_progression(recipient_id, extracted, report_id, extracted.get("report_date", ""), db)
        generate_alerts(recipient_id, progression, db)
        check_monitoring_gaps(recipient_id, db)
        calculate_risk_score(recipient_id, db)

        from tables.users import CareRecipient
        rec = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
        if rec:
            rec.last_report_date = datetime.datetime.utcnow()

        report.processing_status = ReportProcessingStatus.completed
        db.commit()
        print(f"[pipeline_v1] ✅ Report {report_id} complete (legacy mode)")
        try:
            _recalculate_aggregate_summary(db, recipient_id)
        except Exception:
            pass
    except Exception as e:
        print(f"[pipeline_v1] ❌ Error: {e}")
        db.rollback()
        try:
            rep = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
            if rep:
                rep.processing_status = ReportProcessingStatus.failed
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


def _get_username_from_auth(auth_header: Optional[str]):
    if not auth_header:
        return None
    try:
        parts = auth_header.split()
        if len(parts) != 2:
            return None
        token = parts[1]
        from repository.users import JWTRepo
        decoded = JWTRepo.decode_token(token)
        return decoded.get('sub') if isinstance(decoded, dict) else None
    except Exception:
        return None


def _recalculate_aggregate_summary(db: Session, recipient_id: int):
    """Helper to ensure longitudinal history is ALWAYS up to date after any change."""
    try:
        recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
        if not recipient:
            return

        reports = list_reports_for_recipient(db, recipient_id)
        if not reports:
            recipient.report_summary = None
            db.add(recipient)
            db.commit()
            return

        insights = []
        print(f'[recipients] Recalculating history for {len(reports)} reports...')
        for r in reports:
            if not r.analysis_summary:
                txt = extract_text_from_bytes(r.data or b'', r.mime_type)
                if txt:
                    r.analysis_summary = summarize_text_via_gemini(txt, target_words=150)
                    db.add(r)
            
            if r.analysis_summary:
                insights.append(r.analysis_summary)
        
        if insights:
            longitudinal = summarize_report_insights_via_gemini(insights)
            recipient.report_summary = longitudinal
            db.add(recipient)
            db.commit()
            print(f"[recipients] Longitudinal history updated for recipient {recipient_id}")
    except Exception as e:
        print(f"[recipients] Error during aggregate recalculation: {e}")


@router.post('/recipients/{recipient_id}/reports', response_model=ResponseSchema)
async def upload_medical_report(recipient_id: int, background_tasks: BackgroundTasks, file: UploadFile = File(...), authorization: Optional[str] = Header(None), db: Session = Depends(get_db), request: Request = None):
    # Auth
    # Debug: log some request header info
    try:
        hdrs = dict(request.headers)
        print(f"[recipients] Request headers: Authorization={'present' if hdrs.get('authorization') else 'missing'}, Content-Type={hdrs.get('content-type')}")
    except Exception:
        pass

    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Ensure recipient belongs to caretaker
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    # Basic validation
    content = await file.read()
    max_size = 5 * 1024 * 1024  # 5 MB
    if len(content) > max_size:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large")

    # include application/octet-stream to handle some browsers that don't set a specific mime
    allowed = ['application/pdf', 'image/png', 'image/jpeg', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/octet-stream']
    mime = file.content_type or 'application/octet-stream'
    if mime not in allowed:
        # allow some generic types but prefer strict list
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported file type: {mime}")

    # Save to DB
    try:
        # Debug: log mime and size
        try:
            print(f"[recipients] Incoming upload: recipient_id={recipient_id} filename={file.filename} mime={mime} size={len(content)} bytes")
        except Exception:
            print(f"[recipients] Incoming upload: recipient_id={recipient_id} filename={file.filename} (could not determine size)")

        report = create_medical_report(db, recipient_id, file.filename, mime, content)
        print(f"[recipients] Uploaded report id={report.id} recipient_id={recipient_id} filename={file.filename}")

        # Launch async medical history pipeline (v3)
        report_id = report.id
        if background_tasks:
            background_tasks.add_task(
                _run_medical_history_pipeline,
                recipient_id=recipient_id,
                report_id=report_id,
                filename=file.filename
            )
            print(f"[recipients] Medical history pipeline queued for report {report_id}")

        return ResponseSchema(code=200, status='success', message='Report uploaded', result={'report_id': report.id, 'filename': report.filename, 'processing_status': 'processing'})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")


@router.delete('/recipients/{recipient_id}/reports/{report_id}', response_model=ResponseSchema)
def remove_medical_report(recipient_id: int, report_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Delete a medical report."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    report = db.query(MedicalReport).filter_by(id=report_id, care_recipient_id=recipient_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    try:
        if delete_medical_report(db, report_id):
            # Auto-update summary after deletion
            _recalculate_aggregate_summary(db, recipient_id)
            # Auto-recalculate risk score and analysis after deletion
            try:
                from services.medical_history_ai import calculate_risk_score
                calculate_risk_score(recipient_id, db)
                db.commit()
                print(f"[recipients] Risk score recalculated after report {report_id} deletion")
            except Exception as re:
                print(f"[recipients] Risk recalculation after delete failed: {re}")
            return ResponseSchema(code=200, status='success', message='Report deleted successfully')
        else:
            raise HTTPException(status_code=500, detail="Failed to delete report")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion error: {e}")


@router.get('/recipients/{recipient_id}/reports', response_model=ResponseSchema)
def list_reports(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    reports = list_reports_for_recipient(db, recipient_id)
    print(f"[recipients] Listing {len(reports)} reports for recipient_id={recipient_id} (requested by {username})")
    out = [{
        'id': r.id, 
        'filename': r.filename, 
        'mime_type': r.mime_type, 
        'uploaded_at': r.uploaded_at.isoformat(),
        'analysis_summary': r.analysis_summary,
        'processing_status': r.processing_status.value if r.processing_status else 'unknown',
        'report_date': r.report_date.isoformat() if r.report_date else None,
    } for r in reports]
    return ResponseSchema(code=200, status='success', message='Reports fetched', result={'reports': out})


@router.get('/debug/medical_reports/inspect')
def debug_inspect_reports(recipient_id: Optional[int] = None, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Temporary debug endpoint: returns DB url and counts. Requires authentication."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")
    try:
        import config
        from tables.medical_reports import MedicalReport
        total = db.query(MedicalReport).count()
        recip_count = None
        if recipient_id is not None:
            recip_count = db.query(MedicalReport).filter(MedicalReport.care_recipient_id == recipient_id).count()
        return {
            'db_url': str(getattr(config, 'DATABASE_URL', 'unknown')),
            'engine_url': str(getattr(getattr(config, 'engine', ''), 'url', 'unknown')),
            'total_reports': total,
            'recipient_reports': recip_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug inspect failed: {e}")


@router.post('/recipients/{recipient_id}/summarize', response_model=ResponseSchema)
def summarize_recipient_reports(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Re-run summarization for an existing recipient's uploaded reports and save the summary."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    try:
        _recalculate_aggregate_summary(db, recipient_id)
        # return the updated summary
        db.refresh(recipient)
        return ResponseSchema(code=200, status='success', message='Longitudinal history synthesized', result={'summary': recipient.report_summary or ''})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to summarize reports: {e}")


@router.get('/recipients/{recipient_id}/reports/{report_id}/download')
def download_report(recipient_id: int, report_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Stream an individual report back to the authenticated caretaker if they own the recipient."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    report = db.query(__import__('tables.medical_reports', fromlist=['MedicalReport']).MedicalReport).filter_by(id=report_id, care_recipient_id=recipient_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    data = report.data or b''
    stream = BytesIO(data)
    headers = {
        'Content-Disposition': f'attachment; filename="{report.filename}"'
    }
    return StreamingResponse(stream, media_type=report.mime_type or 'application/octet-stream', headers=headers)


@router.get('/recipients/{recipient_id}/reports/{report_id}/extract_preview', response_model=ResponseSchema)
def extract_preview(recipient_id: int, report_id: int, force_refresh: bool = False, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Return a short preview of extracted text for a given report (debug endpoint)."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    report = db.query(MedicalReport).filter_by(id=report_id, care_recipient_id=recipient_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    try:
        # Check cache if not forcing refresh
        if not force_refresh and report.analysis_summary:
            print(f"[recipients] Returning cached snapshot for report_id={report_id}")
            return ResponseSchema(code=200, status='success', message='Cached preview', result={'extracted_preview': report.analysis_summary})

        extracted = extract_text_from_bytes(report.data or b'', report.mime_type)
        if not extracted:
             return ResponseSchema(code=200, status='success', message='No text found', result={'extracted_preview': 'No textual content could be extracted from this file.'})
             
        # Instead of raw preview, return a clinical snapshot
        print(f"[recipients] Generating clinical snapshot for report_id={report_id} (force_refresh={force_refresh})")
        summary = summarize_text_via_gemini(extracted, target_words=150)
        
        # Save to cache
        report.analysis_summary = summary
        db.add(report)
        db.commit()
        
        return ResponseSchema(code=200, status='success', message='Extract preview', result={'extracted_preview': summary})
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to generate report preview: {e}")


@router.post('/recipients/{recipient_id}/reports/base64', response_model=ResponseSchema)
def upload_report_base64(recipient_id: int, payload: dict, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Accept a JSON payload with base64-encoded file to simplify client uploads.
    Payload: { filename: str, mime_type: str, b64: str }
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Care recipient not found for this user")

    try:
        print(f"[recipients] Received base64 upload request for recipient_id={recipient_id} filename={payload.get('filename')}")
        filename = payload.get('filename')
        mime = payload.get('mime_type') or 'application/octet-stream'
        b64 = payload.get('b64')
        if not filename or not b64:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Missing filename or b64 payload')
        import base64
        data = base64.b64decode(b64)
        report = create_medical_report(db, recipient_id, filename, mime, data)

        # Auto-update summary
        _recalculate_aggregate_summary(db, recipient_id)

        return ResponseSchema(code=200, status='success', message='Report uploaded', result={'report_id': report.id, 'filename': report.filename})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")


@router.post('/recipients', response_model=ResponseSchema)
def add_care_recipient(payload: CareRecipientCreate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Add a new care recipient for the current caretaker."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_recipient = CareRecipient(
        caretaker_id=caretaker.id,
        full_name=payload.full_name,
        email=payload.email,
        phone_number=payload.phone_number,
        age=payload.age,
        gender=payload.gender,
        city=payload.city,
        height=payload.height,
        weight=payload.weight,
        blood_group=payload.blood_group,
        emergency_contact=payload.emergency_contact,
        respiratory_condition_status=payload.respiratory_condition_status
    )
    db.add(new_recipient)
    db.commit()
    db.refresh(new_recipient)

    return ResponseSchema(code=200, status="success", message="Care recipient added", result={"id": new_recipient.id})


@router.patch('/recipients/{recipient_id}', response_model=ResponseSchema)
def update_care_recipient(recipient_id: int, payload: RecipientUpdate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Update care recipient details."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=404, detail="Caretaker not found")
        
    print(f"[recipients] PATCH update attempt: recipient_id={recipient_id}, user={username}, caretaker_id={caretaker.id}")
    
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        print(f"[recipients] PATCH failed: Recipient {recipient_id} not found for caretaker {caretaker.id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    if payload.full_name is not None: recipient.full_name = payload.full_name
    if payload.email is not None: recipient.email = payload.email
    if payload.phone_number is not None: recipient.phone_number = payload.phone_number
    if payload.age is not None: recipient.age = payload.age
    if payload.gender is not None: recipient.gender = payload.gender
    if payload.city is not None: recipient.city = payload.city
    if payload.height is not None: recipient.height = payload.height
    if payload.weight is not None: recipient.weight = payload.weight
    if payload.blood_group is not None: recipient.blood_group = payload.blood_group
    if payload.emergency_contact is not None: recipient.emergency_contact = payload.emergency_contact
    if payload.respiratory_condition_status is not None: recipient.respiratory_condition_status = payload.respiratory_condition_status

    db.commit()
    return ResponseSchema(code=200, status="success", message="Recipient updated")


@router.delete('/recipients/{recipient_id}', response_model=ResponseSchema, name="remove_recipient_account")
def remove_care_recipient_account(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Delete a care recipient and all their data (recordings, reports, analysis)."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    db.delete(recipient)
    db.commit()
    return ResponseSchema(code=200, status="success", message="Recipient removed successfully")


# ═══════════════════════════════════════════════════════════════════════
#  SENSOR FUSION INSIGHTS ENGINE ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

@router.get("/recipients/{recipient_id}/insights")
async def get_patient_insights(
    recipient_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Full Sensor Fusion Insights — aggregates ALL data sources (vitals, audio,
    video, medical reports, environment) and generates cross-domain health
    conclusions using deterministic rules + Gemini AI interpretation.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    try:
        from services.insights_engine import generate_full_insights
        result = generate_full_insights(recipient_id, db)
        return result
    except Exception as e:
        print(f"[insights] Error generating insights: {e}")
        import traceback
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════
#  CARE-RECIPIENT FULL PROFILE (v3)
# ═══════════════════════════════════════════════════════════════════════

@router.get("/care-recipients/{recipient_id}/profile")
async def get_care_recipient_profile(
    recipient_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Returns the complete longitudinal health record for a care recipient.
    Powers the unified Care-Recipient Profile Dashboard.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipient not found")

    from tables.medical_conditions import PatientCondition, ConditionStatus, LabValue, MedicalAlert
    from tables.medical_reports import MedicalReport
    from tables.allergies import Allergy
    from tables.medications import Medication, MedicationHistory
    from tables.vital_signs import VitalSign
    from tables.environment import EnvironmentSensor

    # 1. General Info
    general_info = {
        "id": recipient.id,
        "name": recipient.full_name,
        "age": recipient.age,
        "gender": recipient.gender.value if recipient.gender else None,
        "height": recipient.height,
        "weight": recipient.weight,
        "blood_group": recipient.blood_group,
        "phone_number": recipient.phone_number,
        "email": recipient.email,
        "city": recipient.city,
        "emergency_contact": recipient.emergency_contact,
        "respiratory_condition_status": recipient.respiratory_condition_status,
        "registration_date": str(recipient.registration_date) if recipient.registration_date else None,
        "risk_score": recipient.risk_score,
        "risk_factors": recipient.risk_factors_breakdown.get("factors", []) if isinstance(recipient.risk_factors_breakdown, dict) else [],
        "doctor_remarks": recipient.doctor_remarks
    }

    # 2. Active Conditions
    active_conditions_raw = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status != ConditionStatus.resolved
    ).all()
    active_conditions = [
        {
            "id": c.id,
            "disease_name": c.disease_name,
            "status": c.status.value,
            "severity": c.severity.value if c.severity else None,
            "start_date": str(c.first_detected) if c.first_detected else None
        } for c in active_conditions_raw
    ]

    # 3. Medical History (Resolved)
    medical_history_raw = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status == ConditionStatus.resolved
    ).all()
    medical_history = [
        {
            "id": c.id,
            "disease_name": c.disease_name,
            "resolved_date": str(c.resolved_date) if c.resolved_date else None,
            "start_date": str(c.first_detected) if c.first_detected else None
        } for c in medical_history_raw
    ]

    # 4. Allergies
    allergies_raw = db.query(Allergy).filter(Allergy.care_recipient_id == recipient_id).all()
    allergies = [
        {
            "id": a.allergy_id,
            "allergen": a.allergen,
            "type": a.allergy_type.value if a.allergy_type else None,
            "reaction": a.reaction,
            "severity": a.severity,
            "status": a.status.value if a.status else None
        } for a in allergies_raw
    ]

    # 5. Medications
    meds_raw = db.query(Medication).filter(Medication.care_recipient_id == recipient_id).all()
    active_medications = [
        {
            "id": m.medication_id,
            "medicine_name": m.medicine_name,
            "dosage": m.dosage,
            "frequency": m.frequency,
            "schedule_time": m.schedule_time,
            "start_date": str(m.start_date) if m.start_date else None,
            "end_date": str(m.end_date) if m.end_date else None,
            "duration_days": (m.end_date - m.start_date).days if m.end_date and m.start_date else None,
            "status": m.status.value if m.status else None,
            "current_stock": m.current_stock or 0,
            "doses_per_day": m.doses_per_day or 1,
            "predicted_finish_date": str(datetime.date.today() + datetime.timedelta(days=(m.current_stock or 0) // (m.doses_per_day or 1))) if m.current_stock and m.doses_per_day and m.doses_per_day > 0 else None,
            "auto_order_enabled": getattr(m, 'auto_order_enabled', True),
            "last_auto_order_date": str(m.last_auto_order_date) if getattr(m, 'last_auto_order_date', None) else None
        } for m in meds_raw
    ]

    # 6. Medication History
    med_history_raw = db.query(MedicationHistory).filter(MedicationHistory.care_recipient_id == recipient_id).all()
    medication_history = [
        {
            "id": h.history_id,
            "medicine_name": h.medicine_name,
            "dosage": h.dosage,
            "end_date": str(h.end_date) if h.end_date else None,
            "termination_reason": h.termination_reason
        } for h in med_history_raw
    ]

    # 7. Medical Reports
    reports_raw = db.query(MedicalReport).filter(MedicalReport.care_recipient_id == recipient_id).all()
    medical_reports = [
        {
            "id": r.id,
            "filename": r.filename,
            "uploaded_at": str(r.uploaded_at) if r.uploaded_at else None,
            "report_date": str(r.report_date) if r.report_date else None,
            "status": r.processing_status.value if r.processing_status else None
        } for r in reports_raw
    ]

    # 8. Test Trends (Lab Results)
    labs_raw = db.query(LabValue).filter(LabValue.care_recipient_id == recipient_id).order_by(LabValue.recorded_date).all()
    test_trends = [
        {
            "id": l.id,
            "test_name": l.metric_name,
            "test_value": l.normalized_value,
            "test_unit": l.normalized_unit,
            "test_date": str(l.recorded_date) if l.recorded_date else None,
            "is_abnormal": l.is_abnormal,
            "report_id": l.report_id,
            "reference_low": l.reference_range_low,
            "reference_high": l.reference_range_high
        } for l in labs_raw
    ]

    # 9. Vitals (Recent 30)
    vitals_raw = db.query(VitalSign).filter(VitalSign.care_recipient_id == recipient_id).order_by(VitalSign.recorded_at.desc()).limit(30).all()
    vitals = [
        {
            "recorded_at": str(v.recorded_at) if v.recorded_at else None,
            "heart_rate": v.heart_rate,
            "systolic_bp": v.systolic_bp,
            "diastolic_bp": v.diastolic_bp,
            "oxygen_saturation": v.oxygen_saturation,
            "temperature": v.temperature,
            "sleep_score": v.sleep_score,
            "bmi": v.bmi
        } for v in vitals_raw
    ]
    vitals.reverse() # Chronological

    # 10. Environment (Recent 24 hours or latest 20)
    env_raw = db.query(EnvironmentSensor).filter(EnvironmentSensor.care_recipient_id == recipient_id).order_by(EnvironmentSensor.timestamp.desc()).limit(20).all()
    environment = [
        {
            "timestamp": str(e.timestamp) if e.timestamp else None,
            "temperature_c": e.temperature_c,
            "humidity_percent": e.humidity_percent,
            "aqi": e.aqi
        } for e in env_raw
    ]
    environment.reverse()

    # 11. Alerts (Unread & Recent)
    alerts_raw = db.query(MedicalAlert).filter(MedicalAlert.care_recipient_id == recipient_id).order_by(MedicalAlert.created_at.desc()).limit(15).all()
    alerts = [
        {
            "id": a.id,
            "severity": a.severity.value if a.severity else None,
            "message": a.message,
            "is_read": a.is_read,
            "created_at": str(a.created_at) if a.created_at else None
        } for a in alerts_raw
    ]

    return {
        "status": "success",
        "general_info": general_info,
        "conditions": active_conditions,
        "medical_history": medical_history,
        "allergies": allergies,
        "medications": active_medications,
        "medication_history": medication_history,
        "medical_reports": medical_reports,
        "lab_values": test_trends,
        "vitals": vitals,
        "environment": environment,
        "alerts": alerts
    }


# --- Medical Record Management ---

# Conditions
@router.post("/care-recipients/{recipient_id}/conditions", response_model=ResponseSchema)
async def add_condition(recipient_id: int, data: ConditionInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    # Simple check for recipient
    from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity
    import datetime
    
    cond = PatientCondition(
        care_recipient_id=recipient_id,
        disease_code=data.disease_code or "CUSTOM",
        disease_name=data.disease_name,
        status=ConditionStatus(data.status),
        severity=ConditionSeverity(data.severity) if data.severity else None,
        first_detected=datetime.date.fromisoformat(data.first_detected) if data.first_detected else datetime.date.today(),
        last_updated=datetime.date.today()
    )
    db.add(cond)
    db.commit()
    return {"code": 200, "status": "success", "message": "Condition added", "result": {"id": cond.id}}

@router.patch("/care-recipients/{recipient_id}/conditions/{condition_id}", response_model=ResponseSchema)
async def update_condition(recipient_id: int, condition_id: int, data: ConditionInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.medical_conditions import PatientCondition, ConditionStatus, ConditionSeverity
    cond = db.query(PatientCondition).filter(PatientCondition.id == condition_id, PatientCondition.care_recipient_id == recipient_id).first()
    if not cond: raise HTTPException(404, "Condition not found")
    
    cond.disease_name = data.disease_name
    cond.status = ConditionStatus(data.status)
    if data.severity: cond.severity = ConditionSeverity(data.severity)
    db.commit()
    return {"code": 200, "status": "success", "message": "Condition updated"}


# Medications
@router.post("/care-recipients/{recipient_id}/medications", response_model=ResponseSchema)
async def add_medication(recipient_id: int, data: MedicationInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.medications import Medication, MedicationStatus
    import datetime
    # Parse start_date from request or default to today
    start = datetime.date.today()
    if data.start_date:
        try:
            start = datetime.date.fromisoformat(data.start_date)
        except ValueError:
            pass
    # Compute end_date only if duration_days is set (None = lifetime)
    end = None
    if data.duration_days and data.duration_days > 0:
        end = start + datetime.timedelta(days=data.duration_days)

    med = Medication(
        care_recipient_id=recipient_id,
        medicine_name=data.medicine_name,
        dosage=data.dosage,
        frequency=data.frequency,
        schedule_time=data.schedule_time,
        start_date=start,
        end_date=end,
        status=MedicationStatus(data.status),
        current_stock=data.current_stock or 0,
        doses_per_day=data.doses_per_day or 1,
        auto_order_enabled=data.auto_order_enabled if data.auto_order_enabled is not None else True
    )
    db.add(med)
    db.commit()
    return {"code": 200, "status": "success", "message": "Medication added", "result": {"id": med.medication_id}}

@router.patch("/care-recipients/{recipient_id}/medications/{med_id}", response_model=ResponseSchema)
async def update_medication(recipient_id: int, med_id: int, data: MedicationInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.medications import Medication, MedicationStatus
    import datetime
    med = db.query(Medication).filter(Medication.medication_id == med_id, Medication.care_recipient_id == recipient_id).first()
    if not med:
        raise HTTPException(404, "Medication not found")

    med.medicine_name = data.medicine_name
    med.dosage = data.dosage
    med.frequency = data.frequency
    med.schedule_time = data.schedule_time
    med.status = MedicationStatus(data.status)
    med.current_stock = data.current_stock or 0
    med.doses_per_day = data.doses_per_day or 1
    if data.auto_order_enabled is not None:
        med.auto_order_enabled = data.auto_order_enabled

    # Update start_date if provided
    if data.start_date:
        try:
            med.start_date = datetime.date.fromisoformat(data.start_date)
        except ValueError:
            pass
    if not med.start_date:
        med.start_date = datetime.date.today()

    # Recompute end_date: None = lifetime (never auto-complete)
    if data.duration_days and data.duration_days > 0 and med.start_date:
        med.end_date = med.start_date + datetime.timedelta(days=data.duration_days)
    else:
        med.end_date = None  # lifetime — clear any previous end_date

    db.commit()
    return {"code": 200, "status": "success", "message": "Medication updated"}


@router.patch("/care-recipients/{recipient_id}/medications/{med_id}/auto-order", response_model=ResponseSchema)
async def toggle_auto_order(recipient_id: int, med_id: int, payload: dict, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Toggle auto-reorder via Tata 1mg for a specific medication."""
    from tables.medications import Medication
    med = db.query(Medication).filter(
        Medication.medication_id == med_id,
        Medication.care_recipient_id == recipient_id
    ).first()
    if not med:
        raise HTTPException(404, "Medication not found")
    
    enabled = payload.get("auto_order_enabled")
    if enabled is None:
        raise HTTPException(400, "Missing auto_order_enabled in payload")
    
    med.auto_order_enabled = bool(enabled)
    db.commit()
    return {"code": 200, "status": "success", "message": f"Auto-order {'enabled' if enabled else 'disabled'} for {med.medicine_name}"}



# Allergies
@router.post("/care-recipients/{recipient_id}/allergies", response_model=ResponseSchema)
async def add_allergy(recipient_id: int, data: AllergyInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.allergies import Allergy, AllergyType, AllergyStatus
    allg = Allergy(
        care_recipient_id=recipient_id,
        allergen=data.allergen,
        allergy_type=AllergyType(data.allergy_type),
        reaction=data.reaction,
        severity=data.severity,
        status=AllergyStatus(data.status)
    )
    db.add(allg)
    db.commit()
    return {"code": 200, "status": "success", "message": "Allergy added", "result": {"id": allg.allergy_id}}

@router.patch("/care-recipients/{recipient_id}/allergies/{all_id}", response_model=ResponseSchema)
async def update_allergy(recipient_id: int, all_id: int, data: AllergyInput, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.allergies import Allergy, AllergyType, AllergyStatus
    allg = db.query(Allergy).filter(Allergy.allergy_id == all_id, Allergy.care_recipient_id == recipient_id).first()
    if not allg: raise HTTPException(404, "Allergy not found")
    
    allg.allergen = data.allergen
    allg.allergy_type = AllergyType(data.allergy_type)
    allg.reaction = data.reaction
    allg.severity = data.severity
    allg.status = AllergyStatus(data.status)
    db.commit()
    return {"code": 200, "status": "success", "message": "Allergy updated"}

# Recommendations
@router.get("/care-recipients/{recipient_id}/recommendations", response_model=ResponseSchema)
async def get_recommendations(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    from tables.medical_recommendations import MedicalRecommendation
    recs = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == recipient_id
    ).order_by(MedicalRecommendation.created_at.desc()).limit(15).all()

    # Deduplicate by metric locally to only return the latest instance per metric
    latest_recs = {}
    for r in recs:
        if r.metric not in latest_recs:
            latest_recs[r.metric] = r

    results = []
    for r in latest_recs.values():
        results.append({
            "id": r.id,
            "metric": r.metric,
            "severity": r.severity,
            "message": r.message,
            "trigger_value": r.trigger_value,
            "reference_range": r.reference_range,
            "source": r.source,
            "confidence_score": r.confidence_score,
            "actions": r.actions,
            "created_at": r.created_at.isoformat() if r.created_at else None
        })

    # Sort so critical/high come first, matching UI expectation
    priority = {"critical": 4, "high": 3, "medium": 2, "suggestion": 1, "low": 0}
    results.sort(key=lambda x: priority.get(x["severity"], 0), reverse=True)

    return {"code": 200, "status": "success", "message": "Recommendations retrieved", "result": results}

@router.get("/care-recipients/{recipient_id}/health-status", response_model=ResponseSchema)
async def get_health_status(recipient_id: int, db: Session = Depends(get_db)):
    """Backend endpoint for Architect Rule #11 - State of Health categorical score."""
    from services.recommendation_engine import get_state_of_health
    status = get_state_of_health(recipient_id, db)
    return {"code": 200, "status": "success", "message": "Health status retrieved", "result": status}
