from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List
from pydantic import BaseModel
import os
import requests
from config import get_db
from models.users import ResponseSchema
from tables.users import CareRecipient, CareTaker
from repository.users import UsersRepo
from dotenv import load_dotenv
from datetime import datetime, timedelta

from tables.conversation_history import ProactiveReminder, ConversationMessage, MoodEnum, SenderEnum, TriggerTypeEnum, ReminderTypeEnum, RecurrenceEnum
from services.voice_bot_engine import build_conversation_context, generate_system_prompt, analyze_mood, save_message, check_depression_risk
from services.proactive_triggers import get_pending_triggers, create_default_reminders

load_dotenv()

router = APIRouter(tags=["VoiceBot"])

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

def verify_and_get_caretaker(authorization: str, db: Session):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")
    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return caretaker

class ChatRequest(BaseModel):
    recipient_id: Optional[int] = 1
    text: str
    session_id: str = "default_session"
    trigger_type: str = "user_initiated" # or proactive_checkin, reminder, etc.
    language: str = "en"  # 'en' or 'hi' for Hindi

@router.post('/voice-bot/chat', response_model=ResponseSchema)
async def voice_bot_chat(payload: ChatRequest, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    
    recipient = db.query(CareRecipient).filter(CareRecipient.id == payload.recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        return ResponseSchema(code=404, status="error", message="Recipient not found")
        
    user_text = payload.text
    if not user_text:
        return ResponseSchema(code=400, status="error", message="No text provided")

    # 1. Analyze user mood if it's user initiated
    trigger_enum = TriggerTypeEnum(payload.trigger_type) if payload.trigger_type in [t.value for t in TriggerTypeEnum] else TriggerTypeEnum.user_initiated
    mood_result = {"mood": MoodEnum.neutral, "confidence": 0.5}
    if trigger_enum == TriggerTypeEnum.user_initiated:
        mood_result = analyze_mood(user_text)

    # 2. Save User Message
    save_message(payload.recipient_id, SenderEnum.user, user_text, mood_result["mood"], trigger_enum, payload.session_id, db)

    # 3. Build Context & System Prompt
    context = build_conversation_context(payload.recipient_id, db)
    system_prompt = generate_system_prompt(recipient.full_name, context, payload.language)
    
    # Check depression risk to maybe add an alert in response text
    is_at_risk = check_depression_risk(payload.recipient_id, db)
    
    # 4. Call Gemini
    api_key = os.environ.get('GEMINI_API_KEY')
    api_endpoint = os.environ.get('GEMINI_API_ENDPOINT')
    
    if not api_key:
        return ResponseSchema(code=500, status="error", message="Gemini API key not configured")

    url = f"{api_endpoint}?key={api_key}" if api_endpoint else f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}'

    gemini_payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser says: {user_text}"}]}],
        "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.7}
    }
    
    try:
        resp = requests.post(url, json=gemini_payload, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if 'candidates' in data and data['candidates']:
                ai_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                
                # Save Bot Reply
                save_message(payload.recipient_id, SenderEnum.bot, ai_text, MoodEnum.neutral, trigger_enum, payload.session_id, db)
                
                result_data = {
                    "reply": ai_text,
                    "mood_detected": mood_result["mood"].value,
                    "depression_risk": is_at_risk
                }
                return ResponseSchema(code=200, status="success", message="AI response generated", result=result_data)
        else:
            return ResponseSchema(code=resp.status_code, status="error", message=f"Gemini API failed: {resp.status_code}")
    except Exception as e:
        print(f"[voice_bot] Chat error: {e}")
        return ResponseSchema(code=500, status="error", message=f"Chat error: {str(e)}")

@router.get('/voice-bot/triggers/{recipient_id}', response_model=ResponseSchema)
async def fetch_triggers(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    
    recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        return ResponseSchema(code=404, status="error", message="Recipient not found")
        
    triggers = get_pending_triggers(recipient_id, db)
    return ResponseSchema(code=200, status="success", message="Fetched triggers", result={"triggers": triggers})

class ReminderCreate(BaseModel):
    recipient_id: int
    reminder_type: str # water, food, medicine, exercise, custom
    reminder_text: str
    scheduled_time: str # HH:MM
    recurrence: str = "daily"

@router.post('/voice-bot/reminders', response_model=ResponseSchema)
async def create_reminder(payload: ReminderCreate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    recipient = db.query(CareRecipient).filter(CareRecipient.id == payload.recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
    if not recipient:
        return ResponseSchema(code=404, status="error", message="Recipient not found")
        
    r_type = ReminderTypeEnum(payload.reminder_type) if payload.reminder_type in [t.value for t in ReminderTypeEnum] else ReminderTypeEnum.custom
    recur = RecurrenceEnum(payload.recurrence) if payload.recurrence in [t.value for t in RecurrenceEnum] else RecurrenceEnum.daily
    
    reminder = ProactiveReminder(
        care_recipient_id=payload.recipient_id,
        reminder_type=r_type,
        reminder_text=payload.reminder_text,
        scheduled_time=payload.scheduled_time,
        recurrence=recur
    )
    db.add(reminder)
    db.commit()
    
    return ResponseSchema(code=200, status="success", message="Reminder created successfully")

@router.get('/voice-bot/reminders/{recipient_id}', response_model=ResponseSchema)
async def get_reminders(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    reminders = db.query(ProactiveReminder).filter(ProactiveReminder.care_recipient_id == recipient_id).all()
    
    res = [{
        "id": r.id,
        "type": r.reminder_type.value,
        "text": r.reminder_text,
        "time": r.scheduled_time,
        "recurrence": r.recurrence.value,
        "is_active": r.is_active
    } for r in reminders]
    
    return ResponseSchema(code=200, status="success", message="Fetched reminders", result={"reminders": res})

@router.delete('/voice-bot/reminders/{reminder_id}', response_model=ResponseSchema)
async def delete_reminder(reminder_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    reminder = db.query(ProactiveReminder).filter(ProactiveReminder.id == reminder_id).first()
    if not reminder:
        return ResponseSchema(code=404, status="error", message="Reminder not found")
    
    db.delete(reminder)
    db.commit()
    return ResponseSchema(code=200, status="success", message="Reminder deleted successfully")

@router.get('/voice-bot/mood-trend/{recipient_id}', response_model=ResponseSchema)
async def get_mood_trend(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    
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
        
    is_at_risk = check_depression_risk(recipient_id, db)
    
    return ResponseSchema(code=200, status="success", message="Fetched mood trend", result={"trend": mood_counts, "depression_risk": is_at_risk})

@router.post('/voice-bot/init-reminders/{recipient_id}', response_model=ResponseSchema)
async def init_reminders(recipient_id: int, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    caretaker = verify_and_get_caretaker(authorization, db)
    # Check if they already have reminders
    existing = db.query(ProactiveReminder).filter(ProactiveReminder.care_recipient_id == recipient_id).count()
    if existing > 0:
        return ResponseSchema(code=400, status="error", message="Reminders already initialized")
        
    create_default_reminders(recipient_id, db)
    return ResponseSchema(code=200, status="success", message="Default reminders initialized")
