# Caretaker.ai — Full Agent Runbook

Read this entire file before doing anything. It contains everything needed to start, stop, debug, and work on this project.

---

## What This Project Is

Full-stack elderly care platform for caretakers monitoring elderly recipients.

**Features:** AI voice chatbot (Saathi), fall detection, audio event detection, medical report analysis, vital signs, medication management, real-time alerts.

**Stack:**
- Backend: Python FastAPI — port **8000**
- Frontend: Static HTML/CSS/JS via `npm run dev` — port **3000**
- Database: PostgreSQL 13 — port **5433** (NOT 5432)
- AI: Google Gemini API (via `gemini_client.py`)

---

## CRITICAL: Read Before Starting

### PostgreSQL runs on port 5433, NOT 5432
The `.env` `DATABASE_URL` uses port **5433**. PostgreSQL 18 is also installed on this machine but is unused here.

### cloud-sql-proxy hijacks port 5432
`cloud-sql-proxy.exe` at `C:\Users\ARYAN ANGRAL\Desktop\copy of project for testing\cloud-sql-proxy.exe` belongs to a **different project**. If it's running it will conflict. Kill it first:
```bash
netstat -ano | grep ":5432"
taskkill //PID <pid> //F
```

---

## How to Start the Project (in order)

### Step 1 — Kill cloud-sql-proxy if running
```bash
netstat -ano | grep ":5432"
# if any output: taskkill //PID <pid> //F
```

### Step 2 — Start PostgreSQL 13
```bash
"C:/Program Files/PostgreSQL/13/bin/pg_ctl.exe" start -D "C:/Program Files/PostgreSQL/13/data" -l "C:/Program Files/PostgreSQL/13/data/pg_ctl.log"
```
Expected output: `server started`

### Step 3 — Start Backend (in a persistent window)
```bash
powershell -Command "Start-Process cmd -ArgumentList '/k cd /d \"C:\Users\ARYAN ANGRAL\Desktop\caretakerai\backend\" && \"venv\Scripts\activate.bat\" && python main.py'"
```
Or if already in a persistent terminal:
```bash
cd "C:\Users\ARYAN ANGRAL\Desktop\caretakerai\backend"
source venv/Scripts/activate
python main.py
```
Runs on http://localhost:8000. **Do not close this terminal.**

### Step 4 — Start Frontend (in a persistent window)
```bash
powershell -Command "Start-Process cmd -ArgumentList '/k cd /d \"C:\Users\ARYAN ANGRAL\Desktop\caretakerai\frontend\" && npm run dev'"
```
Or if already in a persistent terminal:
```bash
cd "C:\Users\ARYAN ANGRAL\Desktop\caretakerai\frontend"
npm run dev
```
Runs on http://localhost:3000. **Do not close this terminal.**

### Step 5 — Verify Everything Is Running
```bash
netstat -ano | grep -E ":5433|:8000|:3000"
```
All three should show `LISTENING`.

---

## How to Stop the Project

```bash
# Find and kill backend (8000) and frontend (3000)
netstat -ano | grep -E ":8000|:3000"
taskkill //PID <pid> //F

# Stop PostgreSQL 13
"C:/Program Files/PostgreSQL/13/bin/pg_ctl.exe" stop -D "C:/Program Files/PostgreSQL/13/data"
```

---

## Environment Config (.env)

Located at `caretakerai/.env` — gitignored, never commit this file.

```env
DATABASE_URL=postgresql://postgres:start12@localhost:5433/caretaker
SECRET_KEY=crie
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080

MAIL_USERNAME=projectdevhelpid@gmail.com
MAIL_PASSWORD=zukyormijakivzyo
MAIL_FROM=projectdevhelpid@gmail.com
MAIL_PORT=587
MAIL_SERVER=smtp.gmail.com
MAIL_STARTTLS=True
MAIL_SSL_TLS=False

WEATHER_API_KEY=628d4985109c4f6baa3182527250312
GEMINI_API_KEY=AIzaSyCbXYsGN-W75WeXY9Of4mHmmzs2FS9YvGA
GEMINI_MODEL=gemini-2.5-flash-lite
YOUTUBE_API_KEY=AIzaSyArvAj9LsATB1tKUtmtjaD9BsuEOV7tYgM
```

**Only `GEMINI_MODEL` should ever be changed** (to switch when quota is hit).

---

## Git Config

- Remote: https://github.com/P4L4K/caretaker.ai.git
- Branch: `main`
- Git user: **ItsRiddhi** / riddhigupta2268@gmail.com
- **Do NOT include `Co-Authored-By: Claude` in commits**

```bash
cd "C:\Users\ARYAN ANGRAL\Desktop\caretakerai"
git config user.name "ItsRiddhi"
git config user.email "riddhigupta2268@gmail.com"
git remote set-url origin https://ItsRiddhi:<PAT>@github.com/P4L4K/caretaker.ai.git
```

Generate a new classic PAT (repo scope) from ItsRiddhi's GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic).

Commit workflow:
```bash
git add <specific files>
git commit -m "your message"
git push origin main
```

---

## Architecture

```
caretakerai/
├── backend/
│   ├── main.py                        # FastAPI entry point (port 8000)
│   ├── utils/
│   │   └── gemini_client.py           # ALL Gemini API calls — DO NOT MODIFY
│   ├── routes/
│   │   ├── voice_bot.py               # Chat, reminders, mood trend, favorites
│   │   ├── spotify.py                 # YouTube search (API + scrape fallback + quality rank)
│   │   ├── users.py                   # Auth, profile (returns medications[])
│   │   ├── recipients.py              # Care recipient CRUD + medical report upload
│   │   ├── vitals.py                  # Vital signs
│   │   ├── doctor.py                  # Doctor info
│   │   ├── elderly.py                 # Elderly profile
│   │   ├── emergency.py               # Emergency contacts/alerts
│   │   ├── environment.py             # Environment monitoring
│   │   ├── medical_history.py         # Medical history
│   │   ├── recordings.py              # Audio recordings
│   │   ├── audio_events.py            # Audio event detection
│   │   └── video_monitoring.py        # Video/fall detection
│   ├── services/
│   │   ├── voice_bot_engine.py        # Mood, system prompt, DB context builder
│   │   ├── sentiment_engine.py        # 7-day sentiment trend analysis
│   │   ├── alert_engine.py            # Alert generation
│   │   ├── audio_detection.py         # Audio event classification
│   │   ├── disease_detection.py       # Disease detection from vitals
│   │   ├── disease_progression.py     # Progression tracking
│   │   ├── email_notifications.py     # Email alert sending
│   │   ├── insights_engine.py         # Health insights
│   │   ├── lab_value_extractor.py     # Extract lab values from reports
│   │   ├── medical_history_ai.py      # AI-powered medical history
│   │   ├── notification_scheduler.py  # Background notification scheduler
│   │   ├── proactive_triggers.py      # Proactive alert triggers
│   │   └── report_ingestion.py        # Medical report parsing
│   └── tables/                        # SQLAlchemy ORM models — DO NOT MODIFY
├── frontend/
│   ├── voice_bot.html                 # Main voice assistant UI (Saathi)
│   ├── register.html                  # Registration page
│   └── js/                            # Client-side scripts — DO NOT MODIFY
├── model/                             # Pre-trained ML models — DO NOT DELETE
└── .env                               # Config (gitignored)
```

All Gemini calls use `call_gemini()` from `utils/gemini_client.py`. The model name is read from `GEMINI_MODEL` env var at runtime — no hardcoded model names in production code.

---

## Files You Must NOT Modify

| File/Dir | Reason |
|---|---|
| `backend/utils/gemini_client.py` | Central Gemini client — all services depend on it |
| `backend/tables/` | SQLAlchemy ORM models — changing breaks DB schema |
| `backend/migrations/` | Manual migration scripts — do not run unless explicitly asked |
| `.env` | Credentials — only `GEMINI_MODEL` may be changed |
| `frontend/js/` | Client-side scripts — changes affect all UI behavior |
| `model/` | Pre-trained ML models — do not delete or replace |

---

## Common Errors and Fixes

### Port already in use (Errno 10048)
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)
```
**Fix:**
```bash
netstat -ano | grep ":8000"
taskkill //PID <pid> //F
```
Then restart backend.

### UnicodeEncodeError (emoji in print)
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'
```
**Fix:** Replace the emoji in the offending `print()` call with plain text: `[OK]`, `[ERROR]`, etc.

### Gemini Quota Exceeded (429)
```
[Gemini] QUOTA EXCEEDED for model 'gemini-2.5-flash'
```
**Fix:** Edit `.env`, change `GEMINI_MODEL`, restart backend.
```env
# Try in this order:
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MODEL=gemini-2.0-flash
```
Free tier: 20 req/day per model. Resets at midnight Pacific time.

Test which model has quota:
```bash
curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key=AIzaSyCbXYsGN-W75WeXY9Of4mHmmzs2FS9YvGA" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"hi"}]}]}' | head -5
```
`"candidates"` = working. `429` = quota exceeded.

### PostgreSQL Access Denied
```
System error 5 has occurred. Access is denied.
```
**Fix:** Run as Administrator.

### PostgreSQL Not Running (backend crashes on startup)
```bash
netstat -ano | grep ":5433"
# if no output, start postgres:
"C:/Program Files/PostgreSQL/13/bin/pg_ctl.exe" start -D "C:/Program Files/PostgreSQL/13/data" -l "C:/Program Files/PostgreSQL/13/data/pg_ctl.log"
```

### DB Auth Fails on Port 5432
cloud-sql-proxy is running. Kill it:
```bash
netstat -ano | grep ":5432"
taskkill //PID <pid> //F
```

### Voice Bot shows music instead of AI response
`/api/voice-bot/chat` call failed. Check:
1. Backend running? → start it
2. User logged in on frontend? → log in first
3. Gemini quota? → check backend terminal for `QUOTA EXCEEDED`, switch model

### Registration 422 Error
- Phone must be exactly 10 digits
- `care_recipients` must have ≥1 entry with: name, email, age, gender, phone

### Video Unavailable in iframe
YouTube embed error 101/150. Auto-retried by postMessage listener (3 candidates queued). If all 3 fail, falls back to local music.

### Background process exits immediately
Do NOT use `python main.py &`. Use the persistent window command:
```bash
powershell -Command "Start-Process cmd -ArgumentList '/k cd /d \"C:\Users\ARYAN ANGRAL\Desktop\caretakerai\backend\" && \"venv\Scripts\activate.bat\" && python main.py'"
```

---

## Voice Bot (Saathi) — Key Behaviors

- **Hindi TTS**: Always uses hi-IN voice; prefers Microsoft Heera → Kalpana → Google Hindi; rate 0.88 (elderly-friendly)
- **Story detection**: Runs BEFORE music check; handles Devanagari (कहानी, कथा, किस्सा); auto-detects category (horror/comedy/spiritual/historical)
- **Video handler**: वीडियो चलाओ / motivational video play → YouTube search; separate from music flow
- **YouTube quality algo**: Fetches 10 candidates → `videos.list` for real duration + views → filters short clips (songs ≥3min, stories ≥5min) → sorts by view count → returns top 3
- **Auto-retry on embed error**: postMessage listener catches YouTube error 101/150 → auto-tries next in queue → fallback to local music
- **Medicine card**: Loads from DB via `/api/profile` (medications[] field)
- **Mood trend card**: 7-day histogram from `/api/voice-bot/mood-trend`
- **Sentiment strip**: AI emotional summary above chat, updated per message
- **DB favorites**: पसंदीदा गाना checks DB-saved songs first

---

## Database

- **Host:** localhost:**5433**
- **DB name:** caretaker
- **User:** postgres
- **Password:** start12
- **Connection string:** `postgresql://postgres:start12@localhost:5433/caretaker`

---

## API Endpoints (partial)

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/login` | Login, returns JWT |
| POST | `/api/auth/register` | Register new caretaker |
| GET | `/api/profile` | User profile + medications[] |
| POST | `/api/voice-bot/chat` | Send message to Saathi AI |
| GET | `/api/voice-bot/mood-trend` | 7-day mood histogram |
| GET | `/api/voice-bot/favorites` | DB-saved favorite songs |
| GET | `/api/recipients` | List care recipients |
| POST | `/api/recipients` | Add care recipient |
| GET | `/api/vitals/{recipient_id}` | Vital signs |
| POST | `/api/medical-history/upload` | Upload medical report |
| GET | `/api/youtube/search` | Quality-ranked YouTube search |
