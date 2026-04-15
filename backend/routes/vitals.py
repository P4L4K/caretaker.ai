from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import datetime

from config import get_db
from tables.vital_signs import VitalSign
from tables.users import CareRecipient, CareTaker
from repository.users import UsersRepo, JWTRepo

router = APIRouter(tags=["Vitals"])


def _get_username_from_auth(auth_header: Optional[str]):
    if not auth_header:
        return None
    try:
        parts = auth_header.split()
        if len(parts) != 2:
            return None
        token = parts[1]
        decoded = JWTRepo.decode_token(token)
        return decoded.get('sub') if isinstance(decoded, dict) else None
    except Exception:
        return None


# ── Pydantic schema for ESP8266 POST body ─────────────────────────────────────
class VitalPayload(BaseModel):
    care_recipient_id: int          # which elder the data belongs to
    secret_key: str                 # simple shared secret to prevent random writes

    # Sensor fields – all optional so ESP can send only what it has
    heart_rate: Optional[int] = None        # bpm  (MAX30102)
    oxygen_saturation: Optional[int] = None # %    (MAX30102)
    temperature: Optional[float] = None     # °C → stored as-is (convert in frontend if needed)
    systolic_bp: Optional[int] = None       # mmHg (manual / BP module)
    diastolic_bp: Optional[int] = None      # mmHg


class ManualVitalPayload(BaseModel):
    care_recipient_id: int
    heart_rate: Optional[int] = None          # bpm
    systolic_bp: Optional[int] = None         # mmHg
    diastolic_bp: Optional[int] = None        # mmHg
    oxygen_saturation: Optional[int] = None   # %
    temperature: Optional[float] = None       # °C
    sleep_score: Optional[float] = None       # hours (displayed as sleep)
    weight: Optional[float] = None            # kg  (optional — updates profile weight)


# Hard-coded key shared with ESP8266 (change this to something private)
ESP_SECRET_KEY = "caretaker_esp_2024"


@router.post("/vitals/record")
def record_vitals(payload: VitalPayload, db: Session = Depends(get_db)):
    """
    Called by ESP8266 every N seconds to push live sensor readings.
    Validates the secret key, checks the care_recipient_id exists,
    then inserts a new VitalSign row.
    """

    # 1. Authenticate the device
    if payload.secret_key != ESP_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid device secret key")

    # 2. Verify the care recipient exists
    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == payload.care_recipient_id
    ).first()
    if not recipient:
        raise HTTPException(
            status_code=404,
            detail=f"CareRecipient id={payload.care_recipient_id} not found"
        )

    # 3. Insert new vital sign record
    new_vital = VitalSign(
        care_recipient_id=payload.care_recipient_id,
        heart_rate=payload.heart_rate,
        oxygen_saturation=payload.oxygen_saturation,
        temperature=payload.temperature,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        recorded_at=datetime.datetime.utcnow()
    )
    db.add(new_vital)
    db.commit()
    db.refresh(new_vital)

    return {
        "status": "ok",
        "recorded_id": new_vital.id,
        "recorded_at": new_vital.recorded_at.isoformat()
    }


# ═══════════════════════════════════════════════════════════════════════
#  MANUAL VITALS ENTRY (Feature 1 — Dashboard Log Vitals)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/vitals/manual")
def log_manual_vitals(
    payload: ManualVitalPayload,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Caretaker-authenticated endpoint to manually log daily vitals from
    the dashboard. Computes BMI from height+weight and saves the record.
    Triggers the recommendation engine in the background.
    """
    # 1. Authenticate via JWT
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Verify recipient belongs to caretaker
    recipient = db.query(CareRecipient).filter(
        CareRecipient.id == payload.care_recipient_id,
        CareRecipient.caretaker_id == caretaker.id
    ).first()
    if not recipient:
        raise HTTPException(status_code=404, detail="Care recipient not found for this user")

    # 3. Pull height from profile and compute BMI
    profile_height = recipient.height  # cm, stored in profile
    current_weight = payload.weight or recipient.weight  # use new weight or profile weight

    bmi = None
    if profile_height and current_weight:
        height_m = profile_height / 100.0  # cm → m
        if height_m > 0:
            bmi = round(current_weight / (height_m * height_m), 1)

    # Update profile weight if a new one is provided
    if payload.weight:
        recipient.weight = payload.weight

    # 4. Insert vital sign record
    new_vital = VitalSign(
        care_recipient_id=payload.care_recipient_id,
        heart_rate=payload.heart_rate,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        oxygen_saturation=payload.oxygen_saturation,
        temperature=payload.temperature,
        sleep_score=int(payload.sleep_score) if payload.sleep_score else None,
        weight=current_weight,
        height=profile_height,
        bmi=bmi,
        recorded_at=datetime.datetime.utcnow()
    )
    db.add(new_vital)
    db.commit()
    db.refresh(new_vital)

    print(f"[vitals/manual] ✅ Logged vitals for recipient {payload.care_recipient_id} "
          f"(HR={payload.heart_rate}, BP={payload.systolic_bp}/{payload.diastolic_bp}, "
          f"SpO2={payload.oxygen_saturation}, Temp={payload.temperature}, BMI={bmi})")

    # 5. Trigger recommendation engine in background
    background_tasks.add_task(
        _run_recommendation_after_vitals,
        recipient_id=payload.care_recipient_id
    )

    return {
        "status": "ok",
        "recorded_id": new_vital.id,
        "bmi": bmi,
        "recorded_at": new_vital.recorded_at.isoformat()
    }


def _run_recommendation_after_vitals(recipient_id: int):
    """Background task: run recommendation engine after manual vitals entry."""
    try:
        from config import SessionLocal
        from services.recommendation_engine import run_recommendation_engine
        db = SessionLocal()
        try:
            run_recommendation_engine(recipient_id, db, trigger_type="vitals")
        finally:
            db.close()
    except ImportError:
        print("[vitals/manual] recommendation_engine not yet available — skipping")
    except Exception as e:
        print(f"[vitals/manual] Recommendation engine error: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  VITALS HISTORY (for charts)
# ═══════════════════════════════════════════════════════════════════════

@router.get("/vitals/history/{care_recipient_id}")
def get_vitals_history(
    care_recipient_id: int,
    days: int = 30,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Return vitals history for a care recipient (last N days, default 30)."""
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    vitals = (
        db.query(VitalSign)
        .filter(
            VitalSign.care_recipient_id == care_recipient_id,
            VitalSign.recorded_at >= cutoff
        )
        .order_by(VitalSign.recorded_at.asc())
        .all()
    )

    return {
        "care_recipient_id": care_recipient_id,
        "count": len(vitals),
        "vitals": [
            {
                "recorded_at": v.recorded_at.isoformat() if v.recorded_at else None,
                "heart_rate": v.heart_rate,
                "systolic_bp": v.systolic_bp,
                "diastolic_bp": v.diastolic_bp,
                "oxygen_saturation": v.oxygen_saturation,
                "temperature": v.temperature,
                "sleep_score": v.sleep_score,
                "bmi": v.bmi,
                "weight": v.weight
            }
            for v in vitals
        ]
    }


@router.get("/vitals/latest/{care_recipient_id}")
def get_latest_vitals(care_recipient_id: int, db: Session = Depends(get_db)):
    """Return the most recent vital sign row for a given care recipient."""
    vital = (
        db.query(VitalSign)
        .filter(VitalSign.care_recipient_id == care_recipient_id)
        .order_by(VitalSign.recorded_at.desc())
        .first()
    )
    if not vital:
        raise HTTPException(status_code=404, detail="No vitals found for this recipient")

    return {
        "care_recipient_id": care_recipient_id,
        "heart_rate": vital.heart_rate,
        "oxygen_saturation": vital.oxygen_saturation,
        "temperature": vital.temperature,
        "systolic_bp": vital.systolic_bp,
        "diastolic_bp": vital.diastolic_bp,
        "recorded_at": vital.recorded_at.isoformat() if vital.recorded_at else None
    }
