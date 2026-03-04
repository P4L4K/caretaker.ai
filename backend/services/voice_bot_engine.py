import os
import requests
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from tables.conversation_history import ConversationMessage, ProactiveReminder, SenderEnum, MoodEnum, TriggerTypeEnum
from tables.users import CareRecipient
from tables.vital_signs import VitalSign
from tables.medical_conditions import PatientCondition, LabValue, MedicalAlert
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
        .order_by(desc(VitalSign.timestamp)).first()

    # 5. Active Medical Conditions
    conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.is_active == True
    ).all()

    # 6. Recent Lab Values (Abnormal)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    abnormal_labs = db.query(LabValue).filter(
        LabValue.care_recipient_id == recipient_id,
        LabValue.is_abnormal == True,
        LabValue.date_recorded >= thirty_days_ago
    ).all()

    # 7. Recent Audio Events
    audio_events = db.query(AudioEvent).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.timestamp >= seven_days_ago
    ).all()

    # 8. Active Medical Alerts
    active_alerts = db.query(MedicalAlert).filter(
        MedicalAlert.care_recipient_id == recipient_id,
        MedicalAlert.is_active == True
    ).all()

    # 9. Pending Reminders
    now = datetime.utcnow()
    # Simple check for reminders due soon or active
    pending_reminders = db.query(ProactiveReminder).filter(
        ProactiveReminder.care_recipient_id == recipient_id,
        ProactiveReminder.is_active == True
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
        "vitals": latest_vitals.vital_data if latest_vitals else None,
        "conditions": [{"name": c.condition_name, "severity": c.severity} for c in conditions],
        "abnormal_labs": [{"name": l.test_name, "value": l.value, "unit": l.unit} for l in abnormal_labs],
        "audio_events_count": len(audio_events),
        "active_alerts": [{"type": a.alert_type, "message": a.message, "severity": a.severity} for a in active_alerts],
        "reminders": [{"type": r.reminder_type.value, "text": r.reminder_text, "time": r.scheduled_time} for r in pending_reminders]
    }

def generate_system_prompt(context: dict):
    recip = context.get("recipient", {})
    name = recip.get("name", "User")
    
    prompt = f"You are a gentle, proactive, and highly observant AI companion for an elderly person named {name}. "
    prompt += "You initiate conversations, check on their well-being, and provide emotional support.\n\n"
    
    prompt += "### Context about the user:\n"
    prompt += f"- Age: {recip.get('age', 'Unknown')}\n"
    if recip.get('report_summary'):
        prompt += f"- Health Summary: {recip.get('report_summary')}\n"
    
    conds = context.get("conditions", [])
    if conds:
        prompt += "- Active Conditions: " + ", ".join([f"{c['name']} ({c['severity']})" for c in conds]) + "\n"
        
    vitals = context.get("vitals")
    if vitals:
        # Assuming vitals is a dict or string
        prompt += f"- Latest Vitals: {vitals}\n"
        
    alerts = context.get("active_alerts", [])
    if alerts:
        prompt += "- Active Alerts: " + ", ".join([f"{a['type']}: {a['message']}" for a in alerts]) + "\n"
        prompt += "  *IMPORTANT:* Gently bring up these alerts if relevant to the conversation.\n"
        
    reminders = context.get("reminders", [])
    if reminders:
        prompt += "- Daily Reminders: " + ", ".join([f"{r['text']} at {r['time']}" for r in reminders]) + "\n"
        prompt += "  *IMPORTANT:* Naturally weave relevant reminders into the conversation if the time is appropriate.\n"
        
    moods = context.get("mood_counts", {})
    sad_anxious_count = moods.get("sad", 0) + moods.get("anxious", 0) + moods.get("distressed", 0)
    if sad_anxious_count >= 3:
        prompt += "\n**CRITICAL OBSERVATION:** The user has exhibited signs of sadness or distress recently. Be exceptionally gentle, empathetic, and encouraging. Ask how they are feeling emotionally.\n"

    prompt += "\n### Conversation History:\n"
    history = context.get("history", [])
    # Only keep last 5 for prompt context so we don't blow up context window unnecessarily
    for msg in history[-5:]:
        prompt += f"{msg['sender'].capitalize()}: {msg['text']}\n"
        
    prompt += "\n### Rules:\n"
    prompt += "1. Keep responses concise (2-4 sentences) as they will be read aloud.\n"
    prompt += "2. Be empathetic, encouraging, and patient.\n"
    prompt += "3. If they mention severe pain or emergency, strongly advise them to use the emergency button.\n"
    prompt += "4. Use emojis sparingly but warmly.\n"
    prompt += "5. Do not play doctor, but do gently remind them of their care plan (reminders, conditions).\n"
    
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
