# Agent Setup Guide — Caretaker.ai (Fresh Machine)

Read every section fully before running anything. Steps must be done in order.
This guide is written for an AI agent running on Windows. Each step includes
the exact command, expected output, and a fix for every known failure.

---

## 0. What You Are Setting Up

Full-stack elderly care platform.
- Backend: Python 3.12 + FastAPI on port 8000
- Frontend: Static HTML served by Node.js `serve` on port 3000
- Database: PostgreSQL (port 5433 default, or 5432)
- AI: Google Gemini API (key in .env)
- TTS: Google Cloud Text-to-Speech (google_creds.json required)

Repo: https://github.com/ItsRiddhi/chatbotVoice.git

---

## 1. Verify Prerequisites

Run each check. If a tool is missing, install it before continuing.

```bash
python --version        # Need 3.10 or higher
node --version          # Need 18 or higher
npm --version           # Comes with Node
git --version           # Any version
psql --version          # Need PostgreSQL 13+
```

### If Python is missing
Download and install from https://www.python.org/downloads/
During install: CHECK "Add Python to PATH"

### If Node.js is missing
Download from https://nodejs.org (LTS version)

### If PostgreSQL is missing
Download from https://www.postgresql.org/download/windows/
During install: set password (you'll need it for .env), keep default port 5432 or 5433
After install: confirm service is running:
```cmd
sc query state= all | findstr -i postgres
```
Note the exact service name (e.g. postgresql-x64-17) — you'll need it later.

---

## 2. Clone the Repository

```bash
git clone https://github.com/ItsRiddhi/chatbotVoice.git
cd chatbotVoice
```

Expected: folder `chatbotVoice/` created with project files inside.

---

## 3. Place Private Files (REQUIRED — project will not start without these)

You must receive these two files privately from the project owner:
- `.env`
- `google_creds.json`

Place both in the project root: `chatbotVoice/.env` and `chatbotVoice/google_creds.json`

### Edit .env for this machine
Open `.env` and update two values:

1. The database password (use whatever password you set during PostgreSQL install):
```
DATABASE_URL=postgresql://postgres:YOUR_POSTGRES_PASSWORD@localhost:5433/caretaker
```
If PostgreSQL installed on port 5432 (not 5433), change 5433 → 5432.

2. The Google credentials path (update to match this machine's actual path):
```
GOOGLE_APPLICATION_CREDENTIALS="C:\ACTUAL\PATH\TO\chatbotVoice\google_creds.json"
```

Do NOT change any other values in .env (API keys, model names, etc.).

---

## 4. Install System-Level Dependencies

These are NOT installed by pip. They must be installed at the OS level.

### Tesseract OCR (required for report reading feature)

Download installer: https://github.com/UB-Mannheim/tesseract/wiki
Run installer → install to default path (C:\Program Files\Tesseract-OCR\)
After install, add to system PATH:
```
C:\Program Files\Tesseract-OCR\
```

Verify:
```bash
tesseract --version
```
Expected: version number printed. If "command not found" → PATH not set, fix it.

### Poppler (required for PDF reading feature)

Download: https://github.com/oschwartz10612/poppler-windows/releases
Extract zip → you get a folder like `poppler-24.xx.0/`
Add the `bin/` subfolder to system PATH.

Verify:
```bash
pdftoppm -v
```
Expected: version number. If "command not found" → PATH not set, fix it.

---

## 5. Create the Database

Open pgAdmin or a psql terminal and run:

```sql
CREATE DATABASE caretaker;
```

If using psql from command line:
```bash
psql -U postgres -c "CREATE DATABASE caretaker;"
```
Enter the PostgreSQL password when prompted.

Verify:
```bash
psql -U postgres -l | grep caretaker
```
Expected: `caretaker` in the list.

---

## 6. Set Up Python Backend

```bash
cd backend
python -m venv venv
```

Activate the venv:
```bash
# Windows (bash/git bash):
source venv/Scripts/activate

# Windows (cmd):
venv\Scripts\activate.bat

# Mac/Linux:
source venv/bin/activate
```

Confirm venv is active — your prompt should show `(venv)`.

Install dependencies (this takes 20–40 minutes due to torch + tensorflow):
```bash
pip install -r requirements.txt
```

### If pip install fails with dependency conflicts
```bash
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

### If that also fails
Install problematic packages individually first:
```bash
pip install fastapi==0.68.2 --no-deps
pip install pydantic==2.12.5
pip install -r requirements.txt --ignore-installed fastapi
```

### If torch fails (too large / network timeout)
Install torch separately first from official source:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then re-run `pip install -r requirements.txt`

Verify install succeeded:
```bash
python -c "import fastapi, sqlalchemy, torch, tensorflow; print('All OK')"
```
Expected: `All OK`

---

## 7. Set Up Frontend

```bash
cd ../frontend
npm install
```

Expected: `node_modules/` created. Takes under 30 seconds.

Verify:
```bash
npx serve --version
```
Expected: version number.

---

## 8. First Start — Verify Everything

### Start PostgreSQL (if not already running — requires admin terminal)
```cmd
net start postgresql-x64-17
```
Replace `postgresql-x64-17` with the actual service name from Step 1.
If "Access Denied" → run in Administrator cmd (right-click → Run as administrator).

### Start Backend
Open a terminal and keep it open:
```bash
cd chatbotVoice/backend
source venv/Scripts/activate   # or venv\Scripts\activate.bat
python main.py
```

Wait for:
```
Application startup complete.
```
If you see this, backend is ready on port 8000.

### Start Frontend
Open a SECOND terminal and keep it open:
```bash
cd chatbotVoice/frontend
npm run dev
```
Wait for `Accepting connections at http://localhost:3000`

### Verify both are running
```bash
netstat -ano | grep -E ":8000|:3000"
```
Both should show `LISTENING`.

---

## 9. Known Startup Errors and Fixes

### Port already in use (Errno 10048)
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)
```
Fix:
```bash
netstat -ano | grep ":8000"
taskkill //PID <pid_number> //F
```
Then start backend again.

### UnicodeEncodeError (emoji in terminal)
```
UnicodeEncodeError: 'charmap' codec can't encode character '✅'
```
Fix: find the print statement with the emoji and replace it with plain text like `[OK]`.

### Gemini 429 Quota Exceeded
```
[Gemini] Model 'gemini-2.5-flash' overloaded (HTTP 429)
```
Fix: open `.env`, change `GEMINI_MODEL` to a different model:
```
GEMINI_MODEL=gemini-2.5-flash-lite
```
Available models to try in order:
- gemini-2.5-flash-lite
- gemini-2.0-flash
- gemini-2.0-flash-lite
- gemini-flash-latest
- gemini-2.5-flash-preview-05-20
Restart backend after changing.

### Database connection refused
```
could not connect to server: Connection refused
```
Fix: PostgreSQL is not running. Start it (admin terminal):
```cmd
net start postgresql-x64-17
```

### pytesseract not found / tesseract not in PATH
Fix: confirm Tesseract is installed and its folder is in system PATH.
Test: `tesseract --version` in a NEW terminal (not the one you had open before).

### google_creds.json path error
```
FileNotFoundError: google_creds.json
```
Fix: open `.env`, correct the `GOOGLE_APPLICATION_CREDENTIALS` path.
Note: TTS will be silently disabled if credentials are wrong — the app still works, just no audio.

### Background process dies immediately
Never use `python main.py &` — it dies when the shell closes.
Use a persistent window instead:
```bash
powershell -Command "Start-Process cmd -ArgumentList '/k cd /d \"C:\path\to\chatbotVoice\backend\" && venv\Scripts\activate.bat && python main.py'"
```

---

## 10. Confirm App is Working

1. Open http://localhost:3000 in browser
2. You should see the Caretaker.ai login screen
3. Register a new account or log in
4. Try the voice bot — say "hello" or "kya haal hai"
5. Backend terminal should show Gemini API calls succeeding

---

## Architecture Reference

```
chatbotVoice/
├── backend/
│   ├── main.py                  # FastAPI entry point (port 8000)
│   ├── utils/gemini_client.py   # ALL Gemini API calls go through here
│   ├── routes/                  # API route handlers
│   ├── services/                # Business logic
│   └── tables/                  # SQLAlchemy DB models (do not modify)
├── frontend/                    # Static HTML/CSS/JS (port 3000)
├── model/                       # Pre-trained ML models (do not delete)
├── .env                         # Secrets — not in repo, get privately
└── google_creds.json            # Google Cloud creds — not in repo, get privately
```

## Files the Agent Must NEVER Modify

| File | Reason |
|------|--------|
| `backend/utils/gemini_client.py` | Central Gemini client — all services depend on it |
| `backend/tables/` | ORM models — changes break DB schema |
| `backend/migrations/` | Do not run unless explicitly asked |
| `.env` | Only change `GEMINI_MODEL` to switch Gemini models |
| `model/` | Pre-trained ML models — do not delete or replace |
