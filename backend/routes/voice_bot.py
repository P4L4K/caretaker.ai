from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, List
import os
import requests
from config import get_db
from models.users import ResponseSchema
from tables.users import CareRecipient, CareTaker
from repository.users import UsersRepo
from dotenv import load_dotenv

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

@router.post('/voice-bot/chat', response_model=ResponseSchema)
async def voice_bot_chat(payload: dict, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")

    caretaker = UsersRepo.find_by_username(db, CareTaker, username)
    if not caretaker:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user_text = payload.get("text", "")
    recipient_id = payload.get("recipient_id")
    context = payload.get("context", {})

    if not user_text:
        return ResponseSchema(code=400, status="error", message="No text provided")

    api_key = os.environ.get('GEMINI_API_KEY')
    api_endpoint = os.environ.get('GEMINI_API_ENDPOINT')
    
    if not api_key:
        return ResponseSchema(code=500, status="error", message="Gemini API key not configured")

    if api_endpoint:
        url = f"{api_endpoint}?key={api_key}"
    else:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}'

    recipient_name = "User"
    health_summary = ""
    if recipient_id:
        recipient = db.query(CareRecipient).filter(CareRecipient.id == recipient_id, CareRecipient.caretaker_id == caretaker.id).first()
        if recipient:
            recipient_name = recipient.full_name
            health_summary = recipient.report_summary or ""

    system_prompt = f"""You are a gentle, supportive, and helpful AI companion for an elderly person named {recipient_name}. 
Your goal is to provide companionship, answer health-related questions based on their data, and offer emotional support.
Keep your responses warm, respectful, and easy to understand. Use emojis sparingly but warmly.

Context about the user:
- Recent Health Summary: {health_summary or "No summary available."}
- Current Vitals: {context.get('vitals', 'Not provided')}

Rules:
1. If the user mentions pain, severe discomfort, or an emergency, strongly advise them to contact their caretaker or use the emergency button.
2. If they ask for music, stories, or jokes, respond positively but remind them to tap the specific buttons on their dashboard for those features if they want variety.
3. Be encouraging and patient.
4. Keep responses concise (2-4 sentences) as they will be read aloud.
"""

    try:
        gemini_payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser says: {user_text}"}]}],
            "generationConfig": {"maxOutputTokens": 300, "temperature": 0.7}
        }
        
        print(f"[voice_bot] Calling Gemini API at {url.split('?')[0]}")
        resp = requests.post(url, json=gemini_payload, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            if 'candidates' in data and data['candidates']:
                ai_text = data['candidates'][0]['content']['parts'][0]['text']
                return ResponseSchema(code=200, status="success", message="AI response generated", result={"reply": ai_text.strip()})
        else:
            print(f"[voice_bot] Gemini API failed with status {resp.status_code}: {resp.text}")
            return ResponseSchema(code=resp.status_code, status="error", message=f"Gemini API failed: {resp.status_code}")
        
    except Exception as e:
        print(f"[voice_bot] Chat error: {e}")
        return ResponseSchema(code=500, status="error", message=f"Chat error: {str(e)}")
