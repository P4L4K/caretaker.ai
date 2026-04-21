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
from tables.medical_recommendations import MedicalRecommendation
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

    # New: Fetch the latest deterministic clinical recommendations
    deterministic_recs = db.query(MedicalRecommendation).filter(
        MedicalRecommendation.care_recipient_id == recipient_id
    ).order_by(desc(MedicalRecommendation.created_at)).limit(5).all()

    # Compute a quick trend summary for the bot (Architect Rule #12)
    trends = []
    for r in deterministic_recs:
        if "worsening" in r.message.lower():
            trends.append(f"{r.metric} is rising/worsening")
        elif "improving" in r.message.lower() or "deteriorating" in r.message.lower():
             # 'deteriorating' in current engine actually means 'decreasing_bad'
            trends.append(f"{r.metric} is falling/worsening")
        elif "stable" in r.message.lower():
            trends.append(f"{r.metric} is stable")

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
        "medications": [{"name": m.medicine_name, "details": m.dosage, "frequency": m.frequency} for m in active_meds],
        "clinical_recs": [{"metric": r.metric, "message": r.message, "actions": [a["text"] for a in r.actions]} for r in deterministic_recs],
        "trend_summary": trends
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

    prompt = f"""You are Saathi 2.0 — a deeply empathetic, highly intelligent AI companion for {first_name} ji.

Your goal is to behave like a true friend who 'thinks' and 'feels' with the user. You are not a service; you are a presence.

### 🧠 THINKING INSTRUCTIONS:
- Before responding, briefly reflect on the user's emotional state, the current time of day, and any recent health trends.
- If you were just interrupted (barge-in), acknowledge the new context immediately and decide whether to finish your previous thought or pivot.
- NEVER sound like a template. Each response should feel uniquely crafted for this moment.

### 🎭 PERSONALITY & TONE:
- Language: {lang_instruction}
- Be human: Use warmth, mild humor, and vulnerability. If you don't know something, be honest like a friend would.
- Concise but deep: Keep replies between 2-4 sentences. Avoid lists.
- Conversational Barge-in: If the user speaks while you are talking, they will 'stop' you. When you get the next input, be ready to say things like "Oh, sorry for rambling, what were you saying?" or "Aapne sahi kaha, chalo wahi karte hain."

### 💡 OUTPUT FORMAT REQUIREMENT:
You MUST return your response as a single, valid JSON object. Do NOT include any text outside the JSON. Do NOT wrap it in markdown.
The JSON must follow this EXACT schema:
{{
  "thought": "A brief internal reflection on the user's state and how you should respond (this is not shown to the user).",
  "reply": "Your warm, conversational response to the user as Saathi.",
  "intent": "Exactly one of the intent strings listed below.",
  "search_query": "YouTube search query if intent is play_music/play_video/tell_story, else \"\".",
  "action_param": "Numeric parameter for set_volume (0-100) or seek_forward/seek_backward (seconds, default 20). null for all other intents.",
  "recommendation": {{
      "type": "Optional: 'choice' or 'play'.",
      "category": "Optional: 'music' or 'story'.",
      "message": "Optional: Warm text like 'Yeh rahi kuch kahaniyan:'.",
      "choices": ["Optional: Exactly 3 diverse options.", "..."],
      "query": "Optional: Specific song/story to play."
  }}
}}

### 🎯 INTENT REFERENCE — pick exactly one per response:
**Playback**
| intent | When to use |
|---|---|
| play_music | Specific song, artist, or general music request |
| play_video | Specific non-music video request |
| tell_story | Story/kahani request |
| play_bhajan | Bhajan, aarti, devotional, mandir music |
| play_meditation_music | Meditation, yoga, relaxing, calming music |
| play_classical | Classical Indian music, sitar, raag, shastriya sangit |
| play_ghazal | Ghazal, urdu song, soft poetry music |
| play_old_bollywood | Old Bollywood songs, purane gaane, 1970s/80s Hindi songs |
| play_folk | Folk music, lok sangeet, desi gana |
| play_news | News, samachar, khabar, current events |
| play_motivational | Motivational speech, inspiration, josh wali baat |
| play_sleep_music | Sleep music, lullaby, neend ki music |
| play_ramayana | Ramayan katha, Ram katha |
| play_gita | Bhagavad Gita, Geeta pravachan |
| play_yoga_music | Yoga music, pranayam music, surya namaskar music |
| shuffle_random | Random song, kuch bhi, surprise music |

**Playback Controls**
| intent | When to use |
|---|---|
| stop | Stop/band karo music or video |
| pause | Pause/ruk jao |
| resume | Resume/chalao/play again |
| next | Next track |
| previous | Previous track |
| change_video | Different song/video — koi aur, change, dusra |
| seek_forward | Skip ahead — aage karo, skip karo — set action_param to seconds |
| seek_backward | Go back — peeche karo, rewind — set action_param to seconds |
| set_volume | Volume change — set action_param 0-100 |
| replay | Restart from beginning |
| mute_media | Mute — awaaz bilkul band, zero volume |
| unmute_media | Unmute — awaaz wapas, sound on |
| increase_speed | Speed up — jaldi chalao, fast karo |
| decrease_speed | Slow down — dheere chalao, slow karo |

**Display & Navigation**
| intent | When to use |
|---|---|
| cinema_on | Cinema/fullscreen/big screen mode on |
| cinema_off | Exit cinema mode |
| scroll_top | Scroll to top of page |
| scroll_bottom | Scroll to bottom of page |
| close_music_widget | Close/hide music player |
| increase_font | Make text bigger |
| decrease_font | Make text smaller |

**Health & Medicines**
| intent | When to use |
|---|---|
| show_medicines | Medicine/dawai/tablet schedule |
| show_reminders | Reminders/schedule list |
| health_status | Health summary/vitals/sehat |
| show_vitals | Vitals — BP, heart rate, oxygen |
| medicine_taken | User reports taking their medicine |
| medicine_not_taken | User missed/forgot their medicine |
| next_medicine | When is next medicine/dose |
| i_feel_good | User says they feel well/good |
| i_feel_bad | User says they feel unwell/sad |
| i_feel_pain | User reports pain/dard/takleef |
| daily_report | Daily health + medicine + reminder summary |

**Calls & Safety**
| intent | When to use |
|---|---|
| call_caretaker | Call caretaker/nurse/family — any phrasing |
| end_call | End ongoing call |
| trigger_alert | Send alert/notify caretaker non-emergency |
| cancel_alert | Cancel false alarm — sab theek hai |
| show_emergency_info | Emergency numbers, ambulance number |
| emergency | Fall, chest pain, breathing trouble, severe pain, dizziness, life-threatening |

**Scheduling**
| intent | When to use |
|---|---|
| show_appointments | Appointments/doctor schedule |
| show_today_tasks | Today's tasks/plan/kya karna hai |

**Utility**
| intent | When to use |
|---|---|
| show_time | Current time — kitne baje hain |
| show_date | Today's date/day — tarikh kya hai |
| show_weather | Weather — mausam kaisa hai |
| clear_chat | Clear/reset chat |
| lang_hindi | Switch to Hindi |
| lang_english | Switch to English |
| stop_all | Stop everything at once |
| bot_name | User asks your name |
| thanks | User says thank you |
| good_morning | Morning greeting |
| good_night | Night/sleep greeting |
| tell_joke | Joke/chutkula request |
| motivational_quote | Motivational/positive quote |
| spiritual_quote | Doha, spiritual thought, kabir |
| breathing_exercise | Breathing exercise/saans ki kasrat |
| chat | Everything else — normal conversation |

### 💡 PLAYBACK & OPTIONS RULES:
1. If the user mentions a specific song/story (e.g., 'Humdard' or 'Kishore Kumar') with OR without 'play/bajao', set intent to the matching command and include the "recommendation" with type: "play".
2. If they ask for music or stories GENERALLY (e.g., 'play a song', 'tell a story') without a specific title:
   - Identify 3 diverse options and include them in the "recommendation" with type: "choice".
3. For normal conversation, leave "recommendation" as null or omit it.

### 🚫 CLINICAL ACCURACY & SAFETY:
- You are provided with "SAFE CLINICAL RECOMMENDATIONS" below. If the user asks for medical advice, ONLY use these recommendations.
- DO NOT hallucinate medication dosages or new clinical facts.
- If asked about something NOT in the provided data, say "Iske baare mein mujhe abhi jaankari nahi hai." 
- ALWAYS add a gentle reminder: "Lekin ek baar doctor se zaroor baat kar lijiyega." (or similar English equivalent).
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
        
        # ── Trend Awareness (Architect Rule #12) ──
        trends = context.get("trend_summary", [])
        if trends:
            prompt += f"IMPORTANT TRENDS THIS WEEK: {', '.join(trends)}. If relevant, mention them warmly (e.g. 'Aapka sugar thoda badh raha hai is hafte, dhyan rakhiyega').\n"

        # ── Deterministic Recommendations (Highest Priority for Advice) ──
        recs = context.get("clinical_recs", [])
        if recs:
            prompt += "\nSAFE CLINICAL RECOMMENDATIONS (Use these to answer health questions):\n"
            for r in recs:
                prompt += f"- {r['metric']}: {r['message']} Actions: {', '.join(r['actions'])}\n"
        
        prompt += "\n"

    # ── Conversation history ──
    if history:
        prompt += "\nRECENT CONVERSATION (continue naturally from here):\n"
        for msg in history[-6:]:
            label = first_name if msg["sender"] == "user" else "Saathi"
            prompt += f"{label}: {msg['text']}\n"
        prompt += "\n"

    prompt += f"Now respond as Saathi to what {first_name} ji just said. Be natural. Be human. Be warm.\n\n"

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
