"""Periodic Summary Service — Generates 3-day clinical summaries for doctors.

Aggregates:
- Lab value deltas
- Vital sign trends
- Audio event counts (cough/sneeze)
- AI-driven clinical context
"""

import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from tables.users import CareRecipient
from tables.medical_conditions import LabValue, PatientCondition
from tables.vital_signs import VitalSign
from tables.audio_events import AudioEvent
from utils.gemini_client import call_gemini
from services.email_notifications import send_periodic_doctor_summary


def run_periodic_summaries(db: Session):
    """Scan all recipients and send summaries to doctors if 3 days have passed."""
    recipients = db.query(CareRecipient).filter(CareRecipient.doctor_email != None).all()
    
    print(f"[periodic_summary] Scanning {len(recipients)} recipients for summary eligibility...")
    
    for recipient in recipients:
        now = datetime.datetime.utcnow()
        last_summary = recipient.last_doctor_summary_at
        
        # Check if 3 days (72 hours) have passed since last summary
        if last_summary and (now - last_summary).total_seconds() < (3 * 24 * 3600):
            continue
            
        print(f"[periodic_summary] Generating 3-day summary for {recipient.full_name}")
        
        try:
            summary_html = _generate_summary_content(recipient, db)
            if summary_html:
                success = send_periodic_doctor_summary(
                    recipient.doctor_email,
                    recipient.full_name,
                    summary_html
                )
                if success:
                    recipient.last_doctor_summary_at = now
                    db.commit()
                    print(f"[periodic_summary] ✅ Summary sent for {recipient.full_name}")
        except Exception as e:
            print(f"[periodic_summary] ❌ Failed for {recipient.full_name}: {e}")


def _generate_summary_content(recipient: CareRecipient, db: Session) -> str:
    """Gather 3-day data and use Gemini to generate a professional doctor's handover."""
    three_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=3)
    
    # 1. Get recent lab values
    recent_labs = db.query(LabValue).filter(
        LabValue.care_recipient_id == recipient.id,
        LabValue.recorded_date >= three_days_ago.date()
    ).all()
    
    labs_text = ""
    for l in recent_labs:
        status = "⚠️ ABNORMAL" if l.is_abnormal else "Normal"
        labs_text += f"- {l.metric_name}: {l.normalized_value} {l.normalized_unit} ({status})\n"
        
    # 2. Get recent vitals summary
    recent_vitals = db.query(VitalSign).filter(
        VitalSign.care_recipient_id == recipient.id,
        VitalSign.recorded_at >= three_days_ago
    ).all()
    
    vitals_stats = "No vitals recorded in last 3 days."
    if recent_vitals:
        avg_hr = sum(v.heart_rate for v in recent_vitals if v.heart_rate) / len(recent_vitals)
        avg_spo2 = sum(v.oxygen_saturation for v in recent_vitals if v.oxygen_saturation) / len(recent_vitals)
        vitals_stats = f"Avg HR: {avg_hr:.1f}, Avg SpO2: {avg_spo2:.1f}%, Count: {len(recent_vitals)} readings."

    # 3. Audio symptoms
    try:
        coughs = db.query(func.count(AudioEvent.id)).filter(
            AudioEvent.care_recipient_id == recipient.id,
            AudioEvent.detected_at >= three_days_ago,
            AudioEvent.event_type.ilike('%cough%')
        ).scalar() or 0
    except Exception: coughs = 0

    # 4. Use Gemini for a professional summary
    prompt = f"""You are a clinical transcriptionist. Create a professional, concise 3-day health summary for a doctor.
Patient: {recipient.full_name} (Age {recipient.age})
Active Conditions: {recipient.report_summary or 'None specified'}

LATEST DATA (LAST 3 DAYS):
Labs:
{labs_text or 'None recorded.'}

Vitals Snapshot:
{vitals_stats}

Audio Symptoms:
{coughs} cough events detected.

INSTRUCTIONS:
Return the summary in HTML format (use <div>, <p>, <ul>, <li> tags). 
Format it as a "Clinical Handover". 
Highlight any worsening trends or concerns first.
Keep it under 300 words.
"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2}
    }
    
    from utils.gemini_client import call_gemini
    data = call_gemini(payload, caller="[periodic_summary]")
    
    if data and "candidates" in data:
        html = data["candidates"][0]["content"]["parts"][0]["text"]
        # Basic cleanup of markdown if any
        html = html.replace("```html", "").replace("```", "").strip()
        return html
    
    return None
