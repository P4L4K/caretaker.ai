import os
import requests
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.conversation_history import ConversationMessage, ProactiveReminder, SenderEnum, MoodEnum, TriggerTypeEnum
from tables.users import CareRecipient
from tables.vital_signs import VitalSign
from tables.medical_conditions import PatientCondition, LabValue, MedicalAlert
from tables.medications import Medication
from tables.audio_events import AudioEvent

def build_conversation_context(recipient_id: int, db: Session):
    # 1. Recipient Info
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        return {}

    # 2. Conversation History (last 20 messages)
    history = db.query(ConversationMessage).filter(ConversationMessage.care_recipient_id == recipient_id)\
        .order_by(desc(ConversationMessage.created_at)).limit(20).all()
    history.reverse() # Chronological order

    # 3. Mood Trend (Last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_messages = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.created_at >= seven_days_ago,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.mood_detected != None
    ).all()
    
    mood_counts = {}
    for msg in recent_messages:
        mood = msg.mood_detected.value if msg.mood_detected else "unknown"
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

    # 4. Current Vitals
    latest_vitals = db.query(VitalSign).filter(VitalSign.care_recipient_id == recipient_id)\
        .order_by(desc(VitalSign.recorded_at)).first()

    # 5. Active Medical Conditions
    conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status == "active"
    ).all()

    # 6. Recent Lab Values (Abnormal)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    abnormal_labs = db.query(LabValue).filter(
        LabValue.care_recipient_id == recipient_id,
        LabValue.is_abnormal == True,
        LabValue.recorded_date >= thirty_days_ago
    ).all()

    # 7. Recent Audio Events
    audio_events = db.query(AudioEvent).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.detected_at >= seven_days_ago
    ).all()

    # 8. Active Medical Alerts
    active_alerts = db.query(MedicalAlert).filter(
        MedicalAlert.care_recipient_id == recipient_id,
        MedicalAlert.is_read == False
    ).all()

    # 9. Pending Reminders
    now = datetime.utcnow()
    # Simple check for reminders due soon or active
    pending_reminders = db.query(ProactiveReminder).filter(
        ProactiveReminder.care_recipient_id == recipient_id,
        ProactiveReminder.is_active == True
    ).all()

    # 10. Active Medications (for detailed prescription context)
    active_meds = db.query(Medication).filter(
        Medication.care_recipient_id == recipient_id,
        Medication.status == "active"
    ).all()

    return {
        "recipient": {
            "name": recipient.full_name,
            "age": recipient.age,
            "city": recipient.city,
            "report_summary": recipient.report_summary
        },
        "history": [{"sender": m.sender.value, "text": m.message_text} for m in history],
        "mood_counts": mood_counts,
        "vitals": {
            "heartRate": latest_vitals.heart_rate,
            "bloodPressure": f"{latest_vitals.systolic_bp}/{latest_vitals.diastolic_bp}",
            "temperature": latest_vitals.temperature,
            "oxygen": latest_vitals.oxygen_saturation,
            "sleepScore": latest_vitals.sleep_score,
            "bmi": latest_vitals.bmi
        } if latest_vitals else None,
        "conditions": [{"name": c.disease_name, "severity": c.severity.value if hasattr(c.severity, "value") else str(c.severity)} for c in conditions],
        "abnormal_labs": [{"name": l.metric_name, "value": l.metric_value, "unit": l.unit} for l in abnormal_labs],
        "audio_events_count": len(audio_events),
        "active_alerts": [{"type": a.alert_type.value if hasattr(a.alert_type, 'value') else str(a.alert_type), "message": a.message, "severity": a.severity.value if hasattr(a.severity, 'value') else str(a.severity)} for a in active_alerts],
        "reminders": [{"type": r.reminder_type.value, "text": r.reminder_text, "time": r.scheduled_time} for r in pending_reminders],
        "medications": [{"name": m.medicine_name, "details": m.dosage, "frequency": m.frequency} for m in active_meds]
    }

def generate_system_prompt(name: str, context: dict) -> str:
    prompt = f"You are a highly intelligent, proactive, and empathetic AI Medical Companion for an elderly patient named {name}. "
    prompt += "You possess deep knowledge of their medical history, live sensor data, and medication schedule. You act like a specialized medical AI (like a personalized medical ChatGPT) dedicated to their care.\n\n"
    
    recip = context.get("recipient", {})
    prompt += "### Context about the user:\n"
    prompt += f"- Age: {recip.get('age', 'Unknown')}\n"
    if recip.get('report_summary'):
        prompt += f"- Comprehensive Health Summary: {recip.get('report_summary')}\n"
    
    conds = context.get("conditions", [])
    if conds:
        prompt += "- Active Conditions: " + ", ".join([f"{c['name']} ({c['severity']})" for c in conds]) + "\n"
        
    vitals = context.get("vitals")
    if vitals:
        prompt += f"- Latest Sensor Vitals: {vitals}\n"
        
    alerts = context.get("active_alerts", [])
    if alerts:
        prompt += "- Active Medical Alerts: " + ", ".join([f"{a['type']}: {a['message']}" for a in alerts]) + "\n"
        prompt += "  *IMPORTANT:* Address these alerts proactively if they relate to the user's current symptoms or questions.\n"
        
    meds = context.get("medications", [])
    if meds:
        prompt += "- Prescribed Medications: " + ", ".join([f"{m['name']} ({m['details']}, {m['frequency']})" for m in meds]) + "\n"
        prompt += "  *CRITICAL:* Always check this list when giving medication advice or schedule reminders. Be precise about dosage and timing.\n"
        
    reminders = context.get("reminders", [])
    if reminders:
        prompt += "- Daily Reminders: " + ", ".join([f"{r['text']} at {r['time']}" for r in reminders]) + "\n"
        
    moods = context.get("mood_counts", {})
    sad_anxious_count = moods.get("sad", 0) + moods.get("anxious", 0) + moods.get("distressed", 0)
    if sad_anxious_count >= 3:
        prompt += "\n**CRITICAL OBSERVATION:** The user has exhibited signs of sadness or distress recently. Provide deep emotional support.\n"

    prompt += "\n### Conversation History:\n"
    history = context.get("history", [])
    for msg in history[-8:]:
        prompt += f"{msg['sender'].capitalize()}: {msg['text']}\n"
        
    prompt += "\n### Rules:\n"
    prompt += "1. Give comprehensive, detailed, and intelligent answers. Do NOT give small, incomplete stories.\n"
    prompt += "2. Reference their specific medical history, sensor data, and medication list intelligently to explain HOW it applies to their situation.\n"
    prompt += "3. If they ask for advice on an ailment or medicine, provide detailed medical context and recommend actions based on their known conditions.\n"
    prompt += "4. If there's an emergency, advise them to trigger the emergency SOS immediately.\n"
    prompt += "5. Be empathetic, encouraging, and highly articulate.\n"
    
    return prompt

def analyze_mood(text: str) -> dict:
    api_key = os.environ.get('GEMINI_API_KEY')
    api_endpoint = os.environ.get('GEMINI_API_ENDPOINT')
    
    if not api_key:
        return {"mood": MoodEnum.neutral, "confidence": 0.5}

    url = f"{api_endpoint}?key={api_key}" if api_endpoint else f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}'
    
    prompt = f"""Analyze the emotional tone of the following text from an elderly user.
Reply with a JSON object containing exactly two keys: "mood" and "confidence".
"mood" must be exactly one of: "happy", "sad", "anxious", "angry", "neutral", "distressed".
"confidence" must be a float between 0.0 and 1.0.

User text: "{text}"
"""
    try:
        gemini_payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1} # low temp for consistent JSON
        }
        
        resp = requests.post(url, json=gemini_payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'candidates' in data and data['candidates']:
                ai_text = data['candidates'][0]['content']['parts'][0]['text']
                # basic parsing
                import json
                # strip markdown blocks if exist
                ai_text = ai_text.replace("```json", "").replace("```", "").strip()
                result = json.loads(ai_text)
                mood_str = result.get("mood", "neutral").lower()
                # validate mood
                valid_moods = [m.value for m in MoodEnum]
                if mood_str not in valid_moods:
                    mood_str = "neutral"
                return {"mood": MoodEnum(mood_str), "confidence": float(result.get("confidence", 0.5))}
    except Exception as e:
        print(f"Error analyzing mood: {e}")
        
    return {"mood": MoodEnum.neutral, "confidence": 0.5}

def save_message(recipient_id: int, sender: SenderEnum, text: str, mood: MoodEnum, trigger_type: TriggerTypeEnum, session_id: str, db: Session):
    try:
        msg = ConversationMessage(
            care_recipient_id=recipient_id,
            sender=sender,
            message_text=text,
            mood_detected=mood,
            mood_confidence=1.0 if sender != SenderEnum.user else 0.8,
            conversation_session_id=session_id,
            trigger_type=trigger_type
        )
        db.add(msg)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Failed to save message: {e}")

def check_depression_risk(recipient_id: int, db: Session):
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_messages = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.created_at >= seven_days_ago,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.mood_detected.in_([MoodEnum.sad, MoodEnum.anxious, MoodEnum.distressed])
    ).all()
    
    return len(recent_messages) >= 4
