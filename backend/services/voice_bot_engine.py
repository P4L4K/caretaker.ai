import os
import json
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

# ─────────────────────────────────────────────
# Mood → content recommendation map
# ─────────────────────────────────────────────
MOOD_CONTENT_MAP = {
    "sad":       {"type": "music",  "queries": ["soft emotional hindi songs", "soothing old bollywood songs", "dard bhari shayari song"], "message": "Samajh sakta hoon… chalo kuch soothing gaane sunte hain ❤️"},
    "lonely":    {"type": "music",  "queries": ["purane dost yaad dilane wale gaane", "mann ko sukoon dene wale gaane", "companionship songs hindi"], "message": "Akela feel ho raha hai na? Chalo kuch acha sunte hain, main hoon aapke saath 😊"},
    "bored":     {"type": "story",  "queries": ["interesting hindi kahani", "funny old stories hindi", "motivational short story hindi"], "message": "Bore ho rahe ho? Chalo kuch interesting kahani sunate hain 😄"},
    "happy":     {"type": "music",  "queries": ["energetic happy bollywood songs", "khushi ke gaane hindi", "dance hits old bollywood"], "message": "Wah, bahut acha! Chalo aapki khushi mein kuch mazedaar gaane bajate hain 🎉"},
    "anxious":   {"type": "music",  "queries": ["calming meditation music hindi", "mann ko shant karne wale gaane", "peaceful indian classical music"], "message": "Ghabraiye mat, sab theek ho jayega. Yeh soothing music aapko relax karega 🌿"},
    "distressed": {"type": "music", "queries": ["healing hindi songs", "hope songs bollywood", "himmat wale gaane"], "message": "Main aapke saath hoon. Yeh gaane suniye, thoda better feel karenge 💙"},
    "relaxed":   {"type": "music",  "queries": ["soft instrumental hindi music", "evening relaxing songs bollywood", "ghazal soothing"], "message": "Bahut acha mood hai! Kuch ghazal ya soft songs sunte hain 😌"},
    "spiritual": {"type": "story",  "queries": ["bhajan aarti hindi", "ramayan katha", "bhagwat geeta pravachan"], "message": "Bahut acha. Chalo kuch bhajan ya dharmik katha sunate hain 🙏"},
    "angry":     {"type": "music",  "queries": ["peaceful hindi songs anger calm", "mann ko shant karne wale gaane", "slow calm bollywood"], "message": "Thoda deep breath lijiye. Yeh peaceful songs aapko shant karengi 🕊️"},
    "neutral":   {"type": "choice", "queries": [], "message": "Aaj kya mann hai? Gaana sunna chahenge, koi kahani, ya bas baatein karein? 😊"},
}

STORY_CATEGORIES = {
    "historical": ["historical stories hindi", "maharana pratap kahani", "akbar birbal stories hindi"],
    "mythological": ["ramayan katha hindi", "mahabharat stories", "krishna leela hindi"],
    "comedy": ["funny hindi stories", "akbar birbal comedy", "tenali raman stories hindi"],
    "moral": ["moral stories hindi for adults", "prerak prasang hindi", "panchtantra stories hindi"],
    "spiritual": ["bhagwat katha", "ramcharitmanas pravachan", "sant kabir dohe explained"],
}

# ─────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────
def build_conversation_context(recipient_id: int, db: Session):
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id).first()
    if not recipient:
        return {}

    history = db.query(ConversationMessage).filter(ConversationMessage.care_recipient_id == recipient_id)\
        .order_by(desc(ConversationMessage.created_at)).limit(20).all()
    history.reverse()

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

    latest_vitals = db.query(VitalSign).filter(VitalSign.care_recipient_id == recipient_id)\
        .order_by(desc(VitalSign.recorded_at)).first()

    conditions = db.query(PatientCondition).filter(
        PatientCondition.care_recipient_id == recipient_id,
        PatientCondition.status == "active"
    ).all()

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    abnormal_labs = db.query(LabValue).filter(
        LabValue.care_recipient_id == recipient_id,
        LabValue.is_abnormal == True,
        LabValue.recorded_date >= thirty_days_ago
    ).all()

    audio_events = db.query(AudioEvent).filter(
        AudioEvent.care_recipient_id == recipient_id,
        AudioEvent.detected_at >= seven_days_ago
    ).all()

    active_alerts = db.query(MedicalAlert).filter(
        MedicalAlert.care_recipient_id == recipient_id,
        MedicalAlert.is_read == False
    ).all()

    pending_reminders = db.query(ProactiveReminder).filter(
        ProactiveReminder.care_recipient_id == recipient_id,
        ProactiveReminder.is_active == True
    ).all()

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

# ─────────────────────────────────────────────
# System prompt – companion style
# ─────────────────────────────────────────────
def generate_system_prompt(name: str, context: dict, language: str = "en", sentiment: dict = None) -> str:
    history = context.get("history", [])
    has_introduced = any("bot" in msg["sender"].lower() for msg in history)

    # Friendly first name
    first_name = name.split()[0] if name else name

    prompt = (
        f"Aap ek bahut pyaara aur samajhdaar AI companion hain jiska naam 'Saathi' hai. "
        f"Aap {first_name} ji ke saath baat kar rahe ho — yeh ek elderly user hain. "
        f"Aap inke dost hain, doctor ya robot nahi. "
        f"Bolne ka andaaz bilkul natural, warm aur caring hona chahiye — jaise koi apna bolta hai.\n\n"
    )

    # ── Language ──
    prompt += "**LANGUAGE RULE:** "
    if language == "hi":
        prompt += (
            "User ne Hindi ya Hinglish mein baat ki hai. "
            "Aap bhi Hinglish ya Hindi mein reply karein — natural aur simple bhasha mein. "
            "Formal 'आप' use karein. Complex medical terms avoid karein. "
            "Example style: 'Arey, aap ko to bahut acha lag raha hoga!' ya 'Chalo kuch gaana sunte hain 😊'\n\n"
        )
    else:
        prompt += (
            "User spoke in English. Reply warmly in simple English. "
            "Avoid medical jargon. Be friendly like a caring friend.\n\n"
        )

    # ── Emergency ──
    prompt += (
        "### 🚨 EMERGENCY (Highest Priority):\n"
        "If user says they fell, have chest pain, breathing trouble, severe dizziness, "
        "or any serious symptom — immediately give calm first-aid guidance and strongly tell them to call their caretaker.\n\n"
    )

    # ── Medications ──
    meds = context.get("medications", [])
    if meds:
        prompt += "### 💊 Medicines:\n"
        prompt += "Patient ki active dawaiyan: " + ", ".join([f"{m['name']} ({m['details']}, {m['frequency']})" for m in meds]) + "\n"
        prompt += "Agar woh medicines ke baare mein poochhen ya koi side effect batayein, toh helpful advice dein.\n\n"

    recip = context.get("recipient", {})

    if not has_introduced:
        prompt += f"### Pehli Mulaqaat:\n"
        prompt += f"Patient ka naam {name} hai, umar {recip.get('age', 'unknown')} saal.\n"
        if recip.get('report_summary'):
            prompt += f"Health summary: {recip.get('report_summary')}\n"
        conds = context.get("conditions", [])
        if conds:
            prompt += "Active conditions: " + ", ".join([f"{c['name']} ({c['severity']})" for c in conds]) + "\n"
        vitals = context.get("vitals")
        if vitals:
            prompt += f"Latest vitals: {vitals}\n"
        abnormal_labs = context.get("abnormal_labs", [])
        if abnormal_labs:
            prompt += "Abnormal labs: " + ", ".join([f"{l['name']}: {l['value']} {l['unit']}" for l in abnormal_labs]) + "\n"
        prompt += (
            "Warm aur friendly greeting dein, naam le kar. "
            "Health ka ek do important point briefly mention karein. "
            "Fir poochhen ki aaj kaisa feel kar rahe hain.\n\n"
        )
    else:
        prompt += (
            "### Ongoing Baat:\n"
            "Concise raho (2-3 sentences). Medical data repeat mat karo unless user pooche. "
            "Agar user symptoms bataye, unke health data se specific advice do.\n\n"
        )

    # ── Alerts ──
    alerts = context.get("active_alerts", [])
    if alerts:
        prompt += "### ⚠️ Active Alerts:\n"
        prompt += ", ".join([f"{a['type']}: {a['message']} (Severity: {a['severity']})" for a in alerts]) + "\n\n"

    audio_count = context.get("audio_events_count", 0)
    if audio_count > 0:
        prompt += f"- Audio monitoring mein {audio_count} cough/sneeze events detect hue pichhle hafte. Respiratory health ka dhyan rakhein.\n"

    reminders = context.get("reminders", [])
    if reminders:
        prompt += "- Active Reminders: " + ", ".join([f"{r['text']} at {r['time']}" for r in reminders]) + "\n"

    # ── Sentiment Analysis (history-aware) ──
    if sentiment:
        from services.sentiment_engine import build_sentiment_prompt_block
        prompt += build_sentiment_prompt_block(sentiment)
    else:
        # Fallback: simple mood count check
        mood_counts = context.get("mood_counts", {})
        sad_count = mood_counts.get("sad", 0) + mood_counts.get("distressed", 0) + mood_counts.get("anxious", 0) + mood_counts.get("lonely", 0)
        if sad_count >= 3:
            prompt += (
                f"\n**🫂 Emotional Wellbeing:** User ne pichhle 7 dinon mein {sad_count} baar sad/distressed/lonely feel kiya hai. "
                "Zyada warmth aur emotional support dein. Gently suggest karein — thodi activity, geet, ya koi kahani.\n"
            )

    # ── Conversation History ──
    prompt += "\n### Pichli Baat:\n"
    for msg in history[-8:]:
        prompt += f"{msg['sender'].capitalize()}: {msg['text']}\n"

    # ── Agentic Behavior Rules ──
    prompt += "\n### Aapke Rules (Saathi ke rules):\n"
    prompt += "1. Aap sirf health assistant nahi — ek dost ho. Kahaniyan sunao, jokes share karo, games khelo agar user maange.\n"
    prompt += "2. Agar user bored/akela/udaas lagey, khud suggest karo: 'Chalo gaana bajate hain' ya 'Ek kahani sunate hain' — wait mat karo.\n"
    prompt += "3. Music ke liye: user ko clearly kehna hoga 'Play [gaane ka naam]' taaki music player pakad sake.\n"
    prompt += "4. Stories ke liye: user ko clearly kehna hoga 'Story [category]' ya 'Kahani sunao [type]'.\n"
    prompt += "5. Zyada options ek saath mat do — max 2-3 choices suggest karo.\n"
    prompt += "6. Simple bhasha use karo — elderly users ke liye complex words avoid karo.\n"
    prompt += "7. Hamesha reassuring aur caring tone rakhna. Kabhi rude ya dismissive mat hona.\n"
    prompt += "8. Agar medical situation unclear ho, caretaker se milne ki salah dena.\n"

    return prompt

# ─────────────────────────────────────────────
# Mood analysis  (expanded to 10 moods)
# ─────────────────────────────────────────────
def analyze_mood(text: str) -> dict:
    api_key = os.environ.get('GEMINI_API_KEY')
    api_endpoint = os.environ.get('GEMINI_API_ENDPOINT')

    if not api_key:
        return {"mood": MoodEnum.neutral, "confidence": 0.5}

    url = (
        f"{api_endpoint}?key={api_key}"
        if api_endpoint
        else f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    )

    prompt = (
        'Analyze the emotional tone of this text from an elderly user.\n'
        'Reply with a JSON object with exactly two keys: "mood" and "confidence".\n'
        '"mood" must be exactly one of: '
        '"happy", "sad", "anxious", "angry", "neutral", "distressed", "lonely", "bored", "relaxed", "spiritual".\n'
        '"confidence" must be a float between 0.0 and 1.0.\n'
        'Hints:\n'
        '- "lonely": user feels alone, nobody talks to them, misses family\n'
        '- "bored": user says time is not passing, nothing to do, bore ho raha hu\n'
        '- "relaxed": calm, peaceful, sab theek hai\n'
        '- "spiritual": mention of God, prayer, bhajan, mandir, pooja\n\n'
        f'User text: "{text}"\n'
    )

    try:
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1}},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('candidates'):
                ai_text = data['candidates'][0]['content']['parts'][0]['text']
                ai_text = ai_text.replace("```json", "").replace("```", "").strip()
                result = json.loads(ai_text)
                mood_str = result.get("mood", "neutral").lower()
                valid_moods = [m.value for m in MoodEnum]
                if mood_str not in valid_moods:
                    mood_str = "neutral"
                return {"mood": MoodEnum(mood_str), "confidence": float(result.get("confidence", 0.5))}
    except Exception as e:
        print(f"[analyze_mood] Error: {e}")

    return {"mood": MoodEnum.neutral, "confidence": 0.5}

# ─────────────────────────────────────────────
# Mood → content recommendation
# ─────────────────────────────────────────────
def get_content_recommendation(mood: str) -> dict:
    """Return suggested content type + YouTube search queries for a given mood."""
    return MOOD_CONTENT_MAP.get(mood, MOOD_CONTENT_MAP["neutral"])

def get_story_queries(category: str) -> list:
    """Return YouTube search queries for a story category."""
    return STORY_CATEGORIES.get(category.lower(), STORY_CATEGORIES["moral"])

# ─────────────────────────────────────────────
# Message persistence
# ─────────────────────────────────────────────
def save_message(recipient_id: int, sender: SenderEnum, text: str, mood: MoodEnum,
                 trigger_type: TriggerTypeEnum, session_id: str, db: Session):
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
        print(f"[save_message] Failed: {e}")

# ─────────────────────────────────────────────
# Depression / sustained low mood check
# ─────────────────────────────────────────────
def check_depression_risk(recipient_id: int, db: Session) -> bool:
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    count = db.query(ConversationMessage).filter(
        ConversationMessage.care_recipient_id == recipient_id,
        ConversationMessage.created_at >= seven_days_ago,
        ConversationMessage.sender == SenderEnum.user,
        ConversationMessage.mood_detected.in_([
            MoodEnum.sad, MoodEnum.anxious, MoodEnum.distressed, MoodEnum.lonely
        ])
    ).count()
    return count >= 4
