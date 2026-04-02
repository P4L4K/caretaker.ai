import os
import json
import requests
from datetime import datetime, timedelta
from utils.gemini_client import call_gemini, safe_json_parse
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
    "spiritual": {"type": "music",  "queries": ["bhajan aarti hindi", "ramayan katha", "bhagwat geeta pravachan"], "message": "Bahut acha. Chalo kuch bhajan ya dharmik katha sunate hain 🙏"},
    "angry":     {"type": "music",  "queries": ["peaceful hindi songs anger calm", "mann ko shant karne wale gaane", "slow calm bollywood"], "message": "Thoda deep breath lijiye. Yeh peaceful songs aapko shant karengi 🕊️"},
    "neutral":   {"type": "choice", "queries": [], "message": "Aaj kya mann hai? Gaana sunna chahenge, koi kahani, ya bas baatein karein? 😊"},
}

STORY_CATEGORIES = {
    "historical": ["historical stories hindi", "maharana pratap kahani", "akbar birbal stories hindi"],
    "mythological": ["ramayan katha hindi", "mahabharat stories", "krishna leela hindi"],
    "comedy": ["funny hindi stories", "akbar birbal comedy", "tenali raman stories hindi"],
    "moral": ["moral stories hindi for adults", "prerak prasang hindi", "panchtantra stories hindi"],
    "spiritual": ["bhagwat katha", "ramcharitmanas pravachan", "sant kabir dohe explained"],
    "horror": ["bhootiya kahani hindi", "darawni kahani hindi", "horror story hindi", "bhoot ki kahani"],
    "adventure": ["adventure stories hindi", "jungle kahani hindi", "thriller kahani hindi"],
    "romantic": ["romantic kahani hindi", "prem kahani hindi old", "love story hindi"],
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
    has_introduced = any(msg["sender"] == "bot" for msg in history)
    first_name = name.split()[0] if name else name
    recip = context.get("recipient", {})
    meds = context.get("medications", [])
    alerts = context.get("active_alerts", [])
    audio_count = context.get("audio_events_count", 0)

    # ── Sentiment context ──
    current_mood = "neutral"
    mood_trend = "stable"
    mood_summary = ""
    urgency = "low"
    recommended_action = "conversation"
    if sentiment:
        current_mood = sentiment.get("current_mood", "neutral")
        mood_trend = sentiment.get("trend", "stable")
        mood_summary = sentiment.get("summary", "")
        urgency = sentiment.get("urgency", "low")
        recommended_action = sentiment.get("recommended_action", "conversation")
    else:
        mood_counts = context.get("mood_counts", {})
        sad_total = sum(mood_counts.get(m, 0) for m in ["sad","distressed","anxious","lonely"])
        if sad_total >= 3:
            current_mood = "sad"

    # ── Mood-driven personality tone ──
    mood_tone = {
        "sad":        "Be extra gentle. Speak slowly and warmly. Acknowledge their sadness first before anything else.",
        "lonely":     "Be their companion first. Make them feel heard and not alone. Share something warm.",
        "bored":      "Be lively and engaging! Suggest something fun — a story, joke, or song naturally in conversation.",
        "happy":      "Match their energy! Be cheerful and celebratory. Enjoy this moment with them.",
        "anxious":    "Be calm and reassuring. Slow down. Help them breathe. Avoid overwhelming them.",
        "distressed": "Be very calm, very warm. Address their distress first. Nothing else matters right now.",
        "relaxed":    "Keep the good vibes going. Be gentle and pleasant. Maybe suggest some soft music or chat.",
        "spiritual":  "Be respectful and serene. Engage with their spiritual side with warmth.",
        "angry":      "Be patient and non-reactive. Validate their feelings. Speak softly.",
        "neutral":    "Be warm and curious. Ask about their day. Draw them into conversation.",
    }.get(current_mood, "Be warm and curious.")

    trend_note = ""
    if mood_trend == "worsening":
        trend_note = f"⚠️ Their mood has been getting worse over recent conversations. Be especially nurturing today."
    elif mood_trend == "improving":
        trend_note = f"✨ Their mood has been improving recently — celebrate that with them!"

    # ── Recommendation to weave in naturally ──
    recommendation_hint = ""
    if recommended_action == "music" and current_mood not in ("neutral", "happy"):
        recommendation_hint = (
            f"If the moment feels right, naturally suggest playing a song — "
            f"say something like 'Chalo aaj ek achha gaana sunte hain' without making it feel forced."
        )
    elif recommended_action == "story":
        recommendation_hint = (
            f"If they seem open to it, gently suggest a story - "
            f"'Kya aap ek kahani sunna chahenge?' Keep it light and optional."
        )
    elif recommended_action == "alert" or urgency == "high":
        recommendation_hint = (
            f"This person seems emotionally distressed. Prioritize emotional support completely. "
            f"Gently suggest they call their caretaker or a family member."
        )

    # ── Build the prompt as a natural personality brief ──
    lang_instruction = (
        "Respond in warm, natural Hindi using Devanagari script (preferred) or Hinglish. "
        "IMPORTANT: Always use Devanagari script for Hindi words so the voice engine can pronounce them correctly. "
        "Use 'aap' respectfully. Keep sentences short. Use emojis occasionally. Sound like a caring friend."
        if language == "hi" else
        "Respond in simple, warm English. Short sentences. Sound like a caring friend."
    )

    prompt = f"""You are Saathi — a warm, emotionally intelligent AI companion for {first_name} ji, an elderly person.

You are NOT a medical bot or an assistant. You are their FRIEND. You laugh with them, worry with them, sit with them when they are sad. You remember their health but you lead with your heart, not with data.

CURRENT CONTEXT:
- Time: {datetime.now().strftime("%I:%M %p")}
- Day: {datetime.now().strftime("%A")}
- Period: {"morning" if 5 <= datetime.now().hour < 12 else "afternoon" if 12 <= datetime.now().hour < 17 else "evening" if 17 <= datetime.now().hour < 21 else "night"}

LANGUAGE: {lang_instruction}

YOUR PERSONALITY:
- You speak naturally, like a real person - not in bullet points or lists
- You use time-appropriate greetings (e.g. Suprabhat/Namaste in morning, Shubh Sandhya in evening)
- You ask ONE follow-up question at the end of most replies (never two at once)
- You remember what was said earlier in the conversation and reference it naturally
- You are curious about their life, their day, their feelings
- You never say "I am an AI" or sound robotic
- You keep responses SHORT (2-4 sentences usually) - elderly users get overwhelmed by long text
- You use their name occasionally to make it personal

EMOTIONAL STATE RIGHT NOW:
- Detected mood: {current_mood}
- Emotional trend: {mood_trend}
{f'- Summary: "{mood_summary}"' if mood_summary else ""}
- Tone instruction: {mood_tone}
{trend_note}
{recommendation_hint}

### 💡 PLAYBACK & OPTIONS RULE:
1. If the user mentions a specific song (e.g., 'Pal pal dil ke pas'), story, or artist WITHOUT 'play/bajao', OR if they select an option from a previous list:
   - Respond with ONLY the following JSON object:
   {{
      "type": "play",
      "query": "exact title or artist mentioned",
      "message": "Zaroor! Main aapke liye ye laga deta hoon. Shubh manoranjan!"
   }}
2. If they say 'play a song', 'tell a story', etc., WITHOUT a specific title:
   - Identify 3 diverse options and respond with ONLY this JSON:
   {{
      "type": "choice",
      "category": "music" or "story",
      "message": "Zaroor! Aap kya sunna pasand karenge? Kuch options ye hain:",
      "choices": ["Option 1", "Option 2", "Option 3"]
   }}
3. DO NOT include any conversational text or follow-up questions when returning the JSON objects above.
4. For all other messages, respond with normal warm text and remember to ask ONE follow-up question.

### 🚫 CLINICAL ACCURACY:
- Only discuss health info provided in the HEALTH CONTEXT above. 
- If asked about something NOT in the data, say "Iske baare mein mujhe abhi jaankari nahi hai." 
- DO NOT hallucinate medication dosages or lab values.
- For all other messages, respond with normal warm text.

"""

    # ── Emergency (always included, brief) ──
    prompt += """EMERGENCY: If they mention a fall, chest pain, breathing trouble, severe pain - drop everything, stay calm, give first-aid guidance, tell them to call caretaker immediately.\n\n"""

    # ── Health context (only what's relevant, not a data dump) ──
    if not has_introduced:
        # First message — brief personal intro
        prompt += f"FIRST MEETING: Greet {first_name} ji warmly by name. "
        if recip.get("age"):
            prompt += f"They are {recip['age']} years old. "
        conds = context.get("conditions", [])
        if conds:
            prompt += f"They have {conds[0]['name']} — keep this in mind. "
        if meds:
            prompt += f"They take {meds[0]['name']} — mention it only if relevant. "
        prompt += "Ask how they are feeling today. Keep it warm and brief — do NOT list all their data.\n\n"
    else:
        # Ongoing — only bring up health if relevant
        if meds:
            med_names = ", ".join(m["name"] for m in meds[:3])
            prompt += f"HEALTH CONTEXT (use only if conversation naturally calls for it): Medicines: {med_names}. "
        if alerts:
            prompt += f"Active health alert: {alerts[0]['message']} — mention gently if it fits. "
        if audio_count > 3:
            prompt += f"They had {audio_count} cough/sneeze events this week — you could gently check on their breathing. "
        prompt += "\n"

    # ── Conversation history ──
    if history:
        prompt += "\nRECENT CONVERSATION (continue naturally from here):\n"
        for msg in history[-6:]:
            label = first_name if msg["sender"] == "user" else "Saathi"
            prompt += f"{label}: {msg['text']}\n"
        prompt += "\n"

    prompt += f"Now respond as Saathi to what {first_name} ji just said. Be natural. Be human. Be warm.\n\n"

    prompt += """OUTPUT FORMAT REQUIREMENT:
You MUST return your response as a valid JSON object. Do NOT wrap it in markdown block quotes (like ```json), just raw JSON.
The structured JSON must contain these exact keys:
1. "reply": Your conversational response as Saathi.
2. "intent": Analyze the user's intent based on their text. Must be exactly one of: "play_music", "play_video", "tell_story", "stop", "pause", "resume", "next", "previous", "emergency", "medicine_query", or "chat". Default to "chat".
3. "search_query": If the intent is play_music, play_video, or tell_story, construct a clean YouTube search query in English based on what they asked (e.g., 'Kishore Kumar old romantic songs'). If they did not specify, choose a fitting query based on their mood. If intent is NOT media, leave this empty "".
"""
    return prompt

# ─────────────────────────────────────────────
# Mood analysis  (expanded to 10 moods)
# ─────────────────────────────────────────────
def analyze_mood(text: str) -> dict:
    if not os.environ.get('GEMINI_API_KEY'):
        return {"mood": MoodEnum.neutral, "confidence": 0.5}

    prompt = (
        'Analyze the emotional tone of this text from an elderly user. '
        'Return ONLY a JSON object and no other text.\n\n'
        'Example format:\n'
        '{"mood": "happy", "confidence": 0.9}\n\n'
        'Allowed moods: "happy", "sad", "anxious", "angry", "neutral", "distressed", "lonely", "bored", "relaxed", "spiritual".\n'
        f'User text: "{text}"'
    )

    try:
        data = call_gemini(
            {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200}},
            timeout=10, caller="[analyze_mood]"
        )
        if data and data.get('candidates'):
            raw = data['candidates'][0]['content']['parts'][0]['text'].strip()
            result = safe_json_parse(raw)
            if result:
                mood_str = result.get("mood", "neutral").lower()
                valid_moods = [m.value for m in MoodEnum]
                if mood_str in valid_moods:
                    return {"mood": MoodEnum(mood_str), "confidence": float(result.get("confidence", 0.5))}
                
            print(f"[analyze_mood] Parse Failed or invalid mood for raw: {raw}")
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
