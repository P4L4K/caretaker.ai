from fastapi import APIRouter, Depends, File, UploadFile, Header, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Optional, Dict

from models.users import ResponseSchema, RecipientUpdate, CareRecipientCreate
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
async def upload_medical_report(recipient_id: int, file: UploadFile = File(...), authorization: Optional[str] = Header(None), db: Session = Depends(get_db), request: Request = None):
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

        # Auto-update summary
        _recalculate_aggregate_summary(db, recipient_id)

        return ResponseSchema(code=200, status='success', message='Report uploaded', result={'report_id': report.id, 'filename': report.filename})
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
            return ResponseSchema(code=200, status='success', message='Report deleted successfully')
        else:
            raise HTTPException(status_code=500, detail="Failed to delete report")
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
        'analysis_summary': r.analysis_summary
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
