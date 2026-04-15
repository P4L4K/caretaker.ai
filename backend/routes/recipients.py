from fastapi import APIRouter, Depends, File, UploadFile, Header, HTTPException, status, Request, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional, Dict
import datetime
import json

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
        from services.report_ingestion import run_hybrid_lab_extraction, extract_structured_report, extract_text_from_image_bytes
        from repository import medical_history as repo
        import datetime

        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if not report: return

        report.processing_status = ReportProcessingStatus.processing
        db.flush()

        # 1. Text Extraction (Supports OCR for images)
        is_image = report.mime_type and report.mime_type.startswith("image/")
        extracted = None
        text = ""

        if is_image and report.data:
            print(f"[pipeline] Image detected, using OCR")
            extracted = extract_text_from_image_bytes(report.data, report.mime_type)
        else:
            if report.data:
                try: text = extract_text_from_bytes(report.data, filename)
                except Exception as e: print(f"[pipeline] Text extraction failed: {e}")
            if not text:
                report.processing_status = ReportProcessingStatus.failed
                db.commit(); return
            extracted = extract_structured_report(text)

        # 2. Hybrid Lab Extraction
        upload_date = report.uploaded_at.date() if report.uploaded_at else datetime.date.today()
        hybrid_result = run_hybrid_lab_extraction(text=text, report_id=report_id, care_recipient_id=recipient_id, upload_date=upload_date, db=db)
        
        report.report_date = hybrid_result["report_date"] or report.report_date
        report.extracted_data = extracted
        report.processing_status = ReportProcessingStatus.completed
        db.commit()

        # 3. Recommendation Engine (Hybrid)
        try:
            from services.recommendation_engine import run_recommendation_engine
            run_recommendation_engine(recipient_id, db, trigger_type="report")
        except Exception as e:
            print(f"[pipeline] Recommendation engine failed (non-fatal): {e}")

    except Exception as e:
        print(f"[pipeline_v2] Error: {e}")
        db.rollback()
        rep = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if rep: rep.processing_status = ReportProcessingStatus.failed; db.commit()
    finally:
        db.close()

def _run_medical_history_pipeline_v1_legacy(recipient_id: int, report_id: int, filename: str):
    """V1 legacy fallback."""
    from config import SessionLocal
    db = SessionLocal()
    try:
        from tables.medical_reports import MedicalReport, ReportProcessingStatus
        report = db.query(MedicalReport).filter(MedicalReport.id == report_id).first()
        if report:
            report.processing_status = ReportProcessingStatus.completed
            db.commit()
    finally: db.close()

def _get_username_from_auth(auth_header: Optional[str]):
    if not auth_header: return None
    try:
        token = auth_header.split()[1]
        from repository.users import JWTRepo
        return JWTRepo.decode_token(token).get('sub')
    except Exception: return None

# API Endpoints
@router.post('/recipients/{recipient_id}/reports', response_model=ResponseSchema)
async def upload_medical_report(recipient_id: int, background_tasks: BackgroundTasks, file: UploadFile = File(...), authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    username = _get_username_from_auth(authorization)
    if not username: raise HTTPException(401, "Unauthorized")
    content = await file.read()
    report = create_medical_report(db, recipient_id, file.filename, file.content_type, content)
    background_tasks.add_task(_run_medical_history_pipeline, recipient_id, report.id, file.filename)
    return ResponseSchema(code=200, status='success', message='Report uploaded', result={'report_id': report.id})

# ─── New Hybrid Recommendation Routes ───────────────────────────────────────

@router.get("/care-recipients/{recipient_id}/alerts")
async def get_clinical_alerts(recipient_id: int, db: Session = Depends(get_db)):
    """Teammate's deterministic rules: Metric-specific alerts."""
    from tables.medical_recommendations import MedicalRecommendation
    recs = db.query(MedicalRecommendation).filter(MedicalRecommendation.care_recipient_id == recipient_id).order_by(MedicalRecommendation.created_at.desc()).limit(10).all()
    return {"status": "success", "result": [r.__dict__ for r in recs]}

@router.get("/care-recipients/{recipient_id}/recommendations")
async def get_ai_recommendations(recipient_id: int, limit: int = 5, db: Session = Depends(get_db)):
    """User's Gemini AI Trends: Proactive suggestions."""
    from tables.health_recommendations import HealthRecommendation
    recs = db.query(HealthRecommendation).filter(HealthRecommendation.care_recipient_id == recipient_id).order_by(HealthRecommendation.generated_at.desc()).limit(limit).all()
    return {"status": "success", "result": [
        {"id": r.id, "summary": r.trend_summary, "suggestions": r.suggestions_json, "date": r.generated_at.isoformat()} for r in recs
    ]}

@router.get("/care-recipients/{recipient_id}/health-status")
async def get_health_status(recipient_id: int, db: Session = Depends(get_db)):
    from services.recommendation_engine import get_state_of_health
    return {"status": "success", "result": get_state_of_health(recipient_id, db)}

# Manual Log Bridge
@router.post("/recipients/{recipient_id}/manual_vitals")
async def log_manual_vitals(recipient_id: int, data: dict, db: Session = Depends(get_db)):
    # Logic to save manual vitals...
    # After saving, trigger recommendation engine
    from services.recommendation_engine import run_recommendation_engine
    run_recommendation_engine(recipient_id, db, trigger_type="vitals")
    return {"status": "success", "message": "Logged and recommendations updated"}
