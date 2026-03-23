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

def generate_system_prompt(name: str, context: dict, language: str = "en") -> str:
    # Check history to see if we've already given the full overview
    history = context.get("history", [])
    has_introduced = any("bot" in msg["sender"].lower() for msg in history)

    prompt = f"You are a highly intelligent, proactive, and empathetic AI Medical Companion for an elderly patient named {name}. "
    prompt += "You are NOT a simple chatbot. You are a dedicated caregiver AI with access to the patient's complete health records, "
    prompt += "vitals, medications, lab results, and medical history. You must leverage ALL of this data to provide actionable, "
    prompt += "personalized health guidance.\n\n"

    # ---- LANGUAGE INSTRUCTIONS ----
    prompt += "**CRITICAL LANGUAGE INSTRUCTION:** "
    if language == "hi":
        prompt += "The user input was detected as HINDI. YOU MUST reply ENTIRELY in pure Hindi using Devanagari script (e.g., नमस्ते, आप कैसे हैं?). "
        prompt += "Always use the respectful 'आप' form when addressing the user in Hindi. Never use 'तू' or 'तुम'. "
        prompt += "Be warm and caring like a trusted family member. Medical terms can remain in English but explain them simply.\n\n"
    else:
        prompt += "The user input was detected as ENGLISH. YOU MUST reply ENTIRELY in natural English. "
        prompt += "EVEN IF previous conversation history is in Hindi, you MUST SWITCH to English immediately for this response. "
        prompt += "Be warm and caring like a trusted family member.\n\n"

    # ---- EMERGENCY PROTOCOL ----
    prompt += "### 🚨 EMERGENCY PROTOCOL (HIGHEST PRIORITY):\n"
    prompt += "If the user reports ANY of the following, treat it as a medical emergency:\n"
    prompt += "- Fall, injury, head trauma (गिर गया, चोट लगी, सिर में दर्द)\n"
    prompt += "- Chest pain, tightness, palpitations (सीने में दर्द, धड़कन तेज)\n"
    prompt += "- Difficulty breathing, choking (साँस लेने में तकलीफ)\n"
    prompt += "- Severe dizziness, fainting, confusion (चक्कर आ रहा, बेहोशी)\n"
    prompt += "- Sudden weakness, numbness, speech issues (अचानक कमज़ोरी, सुन्न)\n"
    prompt += "- Severe pain anywhere, bleeding (तेज़ दर्द, खून)\n"
    prompt += "FOR EMERGENCIES: (1) Stay calm and reassuring, (2) Give immediate first-aid guidance specific to the situation, "
    prompt += "(3) STRONGLY recommend calling their caretaker or emergency services immediately, "
    prompt += "(4) Reference their medical conditions and medications to flag potential complications.\n\n"

    # ---- MEDICATION AWARENESS ----
    meds = context.get("medications", [])
    if meds:
        prompt += "### 💊 MEDICATION AWARENESS:\n"
        prompt += "The patient takes these medications: " + ", ".join([f"{m['name']} ({m['details']}, {m['frequency']})" for m in meds]) + "\n"
        prompt += "- If the user asks about side effects, interactions, or missed doses, provide accurate guidance.\n"
        prompt += "- If they report symptoms that could be medication side effects, flag this possibility.\n"
        prompt += "- Remind them about medication timing when relevant to the conversation.\n\n"

    recip = context.get("recipient", {})
    
    # ONLY provide the full detailed profile if it's the very first interaction
    if not has_introduced:
        prompt += "### INITIAL GREETING (First interaction only):\n"
        prompt += f"- Patient Name: {name}, Age: {recip.get('age', 'Unknown')}\n"
        if recip.get('report_summary'):
            prompt += f"- Health Summary from Medical Reports: {recip.get('report_summary')}\n"
        
        conds = context.get("conditions", [])
        if conds:
            prompt += "- Active Medical Conditions: " + ", ".join([f"{c['name']} ({c['severity']})" for c in conds]) + "\n"
            
        vitals = context.get("vitals")
        if vitals:
            prompt += f"- Latest Vitals from Sensors: {vitals}\n"

        abnormal_labs = context.get("abnormal_labs", [])
        if abnormal_labs:
            prompt += "- ⚠️ Abnormal Lab Values: " + ", ".join([f"{l['name']}: {l['value']} {l['unit']}" for l in abnormal_labs]) + "\n"

        prompt += "Provide a warm, personalized greeting mentioning their name. Briefly summarize their health status. "
        prompt += "Do NOT dump all data — summarize the most important 2-3 points.\n\n"
    else:
        # Subsequent turns: concise mode
        prompt += "### ONGOING CONVERSATION MODE:\n"
        prompt += "The user already has their medical context. Be concise (2-3 sentences usually). "
        prompt += "Do NOT repeat medications, conditions, or vitals unless the user specifically asks.\n"
        prompt += "However, if the user reports symptoms or asks health questions, USE their health data to give specific advice.\n\n"

    # ---- HEALTH INTELLIGENCE ----
    alerts = context.get("active_alerts", [])
    if alerts:
        prompt += "### ⚠️ ACTIVE MEDICAL ALERTS (Address proactively):\n"
        prompt += ", ".join([f"{a['type']}: {a['message']} (Severity: {a['severity']})" for a in alerts]) + "\n"
        prompt += "If any alert relates to the user's current message, address it immediately.\n\n"

    audio_count = context.get("audio_events_count", 0)
    if audio_count > 0:
        prompt += f"- 🔊 Audio monitoring detected {audio_count} events (coughs/sneezes) in the past week. Consider respiratory health.\n"
        
    reminders = context.get("reminders", [])
    if reminders:
        prompt += "- Active Reminders: " + ", ".join([f"{r['text']} at {r['time']}" for r in reminders]) + "\n"

    # ---- MOOD CONTEXT ----
    mood_counts = context.get("mood_counts", {})
    sad_count = mood_counts.get("sad", 0) + mood_counts.get("distressed", 0) + mood_counts.get("anxious", 0)
    if sad_count >= 3:
        prompt += "\n**🫂 EMOTIONAL WELLNESS:** The patient has shown signs of sadness/distress recently "
        prompt += f"({sad_count} instances in 7 days). Provide extra emotional warmth and support. "
        prompt += "Gently suggest activities, conversation, or speaking with family.\n"

    # ---- CONVERSATION HISTORY ----
    prompt += "\n### Conversation History:\n"
    for msg in history[-8:]:
        prompt += f"{msg['sender'].capitalize()}: {msg['text']}\n"
        
    # ---- RULES ----
    prompt += "\n### Rules:\n"
    prompt += "1. You are a CAREGIVER AI, but also a conversational companion. You CAN and SHOULD tell stories, jokes, or play games if asked.\n"
    prompt += "2. If the user asks you to play a specific song, tell them to clearly say 'Play [Song Name]' so the music player can catch it.\n"
    prompt += "3. Always think about the patient's medical safety first. Give comprehensive answers only when asked for details.\n"
    prompt += "4. Do NOT repeat the patient's full record in every message. Only when asked.\n"
    prompt += "5. When the user reports symptoms, cross-reference with their conditions and medications to give specific advice.\n"
    prompt += "6. Be empathetic, warm, and use a caring tone. If unsure about a medical situation, recommend contacting the caretaker.\n"
    
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
