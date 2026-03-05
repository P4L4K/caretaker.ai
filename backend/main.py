from fastapi import FastAPI, Request, status, UploadFile, File, HTTPException, BackgroundTasks, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session
import subprocess
import sys
import os
import uuid
import shutil
import time
import re
from typing import Dict, Optional, List
import json
from datetime import datetime, timedelta
import psutil
from pathlib import Path
import base64
from fastapi.responses import FileResponse
# Database configuration
# Database configuration
# Use engine and SessionLocal from config to ensure we point to the correct DB (PostgreSQL) defined in .env
from config import engine, SessionLocal, get_db

# Import database models and tables
import tables.users as user_tables
import tables.recordings as recordings_tables
import tables.medical_reports as med_reports_tables
import tables.video_analysis as video_analysis_tables
import tables.vital_signs as vital_signs_tables
import tables.audio_events as audio_events_tables
import tables.medical_conditions as medical_conditions_tables
import tables.disease_dictionary as disease_dictionary_tables
import tables.conversation_history as conversation_history_tables
import tables.environment as environment_tables
import tables.medications as medications_tables
import tables.allergies as allergies_tables

# Import routes
import routes.users as user_routes
import routes.recordings as recordings_routes
import routes.recipients as recipients_routes
import routes.emergency as emergency_routes
from routes import elderly, voice_bot

# Create database tables
user_tables.Base.metadata.create_all(bind=engine)
recordings_tables.Base.metadata.create_all(bind=engine)
med_reports_tables.Base.metadata.create_all(bind=engine)
video_analysis_tables.Base.metadata.create_all(bind=engine)
vital_signs_tables.Base.metadata.create_all(bind=engine)
audio_events_tables.Base.metadata.create_all(bind=engine)
medical_conditions_tables.Base.metadata.create_all(bind=engine)
disease_dictionary_tables.Base.metadata.create_all(bind=engine)
conversation_history_tables.Base.metadata.create_all(bind=engine)
environment_tables.Base.metadata.create_all(bind=engine)
medications_tables.Base.metadata.create_all(bind=engine)
allergies_tables.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CareTaker AI Backend")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Mount the model directory to be served at /api/model
app.mount("/api/model", StaticFiles(directory="../model"), name="model")

@app.on_event("startup")
def ensure_recordings_schema():
    """Ensure columns exist on startup. Uses SQLAlchemy Inspect for DB-agnostic checks."""
    try:
        inspector = inspect(engine)
        
        # 1. recordings.care_recipient_id
        if inspector.has_table("recordings"):
            columns = [c["name"] for c in inspector.get_columns("recordings")]
            if "care_recipient_id" not in columns:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE recordings ADD COLUMN care_recipient_id integer"))
                    print("Added column care_recipient_id to recordings table")

        # 2. care_recipients.report_summary
        if inspector.has_table("care_recipients"):
            columns_cr = [c["name"] for c in inspector.get_columns("care_recipients")]
            if "report_summary" not in columns_cr:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE care_recipients ADD COLUMN report_summary text"))
                    print("Added column report_summary to care_recipients table")
                    
        print("Startup schema check completed.")
    except Exception as e:
        print(f"Startup schema check warning: {e}")

# Serve a static folder (optional) so files like a favicon can be served
static_path = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_path):
    try:
        os.makedirs(static_path, exist_ok=True)
    except Exception:
        pass
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Also mount the repository-level `model` directory (serves model.json, metadata.json, weights.bin)
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sdk', 'model'))
if os.path.exists(model_dir):
    app.mount("/model", StaticFiles(directory=model_dir), name="model")

@app.get("/")
async def root():
    return {"message": "Welcome to CareTaker API", "status": "active"}

# Return a small in-memory favicon to avoid 404s when browsers request it.
# If a real favicon file exists in `backend/static/favicon.ico` it will be served instead.
@app.get('/favicon.ico')
async def favicon():
    ico_path = os.path.join(static_path, 'favicon.ico')
    if os.path.exists(ico_path):
        return FileResponse(ico_path)
    # 1x1 transparent PNG (base64) returned as image/png for simplicity
    png_b64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII='
    return Response(content=base64.b64decode(png_b64), media_type='image/png')

# Include routers with proper prefixes
app.include_router(user_routes.router, prefix="/api")
app.include_router(recordings_routes.router, prefix="/api")
app.include_router(recipients_routes.router, prefix="/api")
app.include_router(emergency_routes.router, prefix="/api")
app.include_router(elderly.router, prefix="/api")
from routes import video_monitoring
app.include_router(video_monitoring.router, prefix="/api")
from routes import audio_events
app.include_router(audio_events.router, prefix="/api")
app.include_router(voice_bot.router, prefix="/api")
from routes import vitals as vitals_routes
app.include_router(vitals_routes.router, prefix="/api")
from routes import medical_history as medical_history_routes
app.include_router(medical_history_routes.router)

from routes import environment as environment_routes
app.include_router(environment_routes.router, prefix="/api")

# Debug: Print all registered routes
print("\n=== Registered Routes ===")
for route in app.routes:
    if hasattr(route, "path") and hasattr(route, "methods"):
        methods = ", ".join(route.methods)
        print(f"{route.path} - {methods}")
print("=======================\n")

# Global exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Global exception handler for HTTP exceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": exc.detail or "Not authenticated"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Global exception handler for all other exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Add this dictionary to track processes
process_status = {}

# Add this endpoint
@app.get("/api/fall-detection/status/{process_id}")
async def get_status(process_id: int):
    if process_id not in process_status:
        raise HTTPException(status_code=404, detail="Process not found")
    
    status = process_status[process_id]
    
    # The error suggests there's a reference to 'time' here that's not defined
    # It should be using time.time() or similar
    if "start_time" in status:
        elapsed = time.time() - status["start_time"]
        status["elapsed"] = round(elapsed, 2)
    
    return status

# Add this with your other routes
@app.post("/api/fall-detection/process-video")
async def process_video( file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    process_id = None
    try:
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_extension}"
        
        # Save the uploaded file
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the video using the unified video monitor script
        # run_video_monitor.py handles output directory creation ("processed_videos")
        # and output filename generation ("analyzed_" + filename)
        output_dir = "processed_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"analyzed_{filename}"
        output_path = os.path.join(output_dir, output_file)

        # Path to the script relative to backend root
        script_path = os.path.join("VideoMonitoring", "run_video_monitor.py")
        
        # Use sys.executable to ensure we use the same python interpreter
        cmd = f'"{sys.executable}" "{script_path}" "{file_path}"'
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Store process info
        process_id = process.pid
        process_status[process_id] = {
            "status": "processing",
            "output_file": output_file,
            "output_path": output_path,
            "progress": 0,
            "error": None
        }
        
        # Add cleanup task
        if background_tasks:
            background_tasks.add_task(
                monitor_process,
                process_id=process_id,
                process=process,
                output_file=output_file
            )
        
        return {
            "process_id": process_id,
            "status": "processing",
            "output_file": output_file
        }
        
    except Exception as e:
        if process_id and process_id in process_status:
            process_status[process_id]["status"] = "error"
            process_status[process_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

# Add this helper function
async def monitor_process(process_id: int, process: subprocess.Popen, output_file: str):
    """
    Monitor the fall detection process and update the status.
    This runs in the background and updates the process status.
    """
    try:
        # Wait for the process to complete
        stdout_data = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                stdout_data.append(output.strip())
                print(output.strip())  # Log output
                
                # Parse progress if available
                if "Progress:" in output:
                    try:
                        progress = int(output.split("Progress:")[1].strip().strip("%"))
                        if process_id in process_status:
                            process_status[process_id]["progress"] = progress
                    except (IndexError, ValueError):
                        pass
        
        # Process completed, update status
        if process_id in process_status:
            if process.returncode == 0:
                process_status[process_id].update({
                    "status": "completed",
                    "progress": 100,
                    "output": "\n".join(stdout_data)
                })
                
                # Parse fall detection results if available
                falls_detected = []
                for i, line in enumerate(stdout_data):
                    if "FALL DETECTED!" in line:
                        try:
                            # Get the time from the next line
                            time_line = stdout_data[i + 1]
                            time_str = time_line.split("Video Time:")[1].strip()
                            
                            # Get angle from the line after next, handling non-ASCII characters
                            if i + 2 < len(stdout_data) and "Angle:" in stdout_data[i + 2]:
                                angle_line = stdout_data[i + 2]
                                # Clean the angle string by removing non-ASCII characters
                                angle_str = angle_line.split("Angle:")[1].split("°")[0].strip()
                                angle = float(angle_str.replace('Â', '').strip())  # Remove non-ASCII characters
                                
                                # Use angle as a proxy for confidence (normalized to 0-1)
                                confidence = min(1.0, angle / 45.0)
                                
                                # Convert time string to seconds
                                h, m, s = map(float, time_str.split(":"))
                                timestamp_seconds = h * 3600 + m * 60 + s
                                
                                falls_detected.append({
                                    "timestamp": time_str,
                                    "timestamp_seconds": timestamp_seconds,
                                    "confidence": confidence,
                                    "angle": angle
                                })
                        except (IndexError, ValueError) as e:
                            print(f"Error parsing fall detection output: {e}")
                
                if falls_detected:
                    process_status[process_id]["falls_detected"] = falls_detected
                    process_status[process_id]["has_falls"] = True
                else:
                    process_status[process_id]["has_falls"] = False
                    
            else:
                process_status[process_id].update({
                    "status": "error",
                    "error": f"Process failed with return code {process.returncode}",
                    "output": "\n".join(stdout_data)
                })
    except Exception as e:
        if process_id in process_status:
            process_status[process_id].update({
                "status": "error",
                "error": str(e)
            })
        print(f"Error in monitor_process: {e}")
    finally:
        # Cleanup temporary files
        try:
            if os.path.exists(f"temp_uploads/{output_file}"):
                os.remove(f"temp_uploads/{output_file}")
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

# Create processed_videos directory if it doesn't exist (used by video_monitoring.py)
processed_videos_dir = os.path.join(os.path.dirname(__file__), "processed_videos")
os.makedirs(processed_videos_dir, exist_ok=True)

# Serve processed videos from the processed_videos directory
# This serves videos at /processed_videos/{filename} for both live monitoring and uploaded analysis
app.mount("/processed_videos", StaticFiles(directory=processed_videos_dir), name="processed_videos")

# Add these imports at the top of the file with other imports
from fastapi import APIRouter, HTTPException
from datetime import datetime
from weather import WeatherPredictionModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Weather router
weather_router = APIRouter()
weather_model = None

@weather_router.get("/weather/current")
async def get_current_weather(recipient_id: Optional[int] = None, db: Session = Depends(get_db)):
    global weather_model
    
    if not weather_model:
        return {
            "error": "Weather service not available",
            "status": "error"
        }
    
    target_city = None
    if recipient_id:
        recipient = db.query(user_tables.CareRecipient).filter(user_tables.CareRecipient.id == recipient_id).first()
        if recipient and recipient.city:
            target_city = recipient.city
    
    try:
        data = weather_model.fetch_data(city=target_city)
        if not data:
            return {
                "error": "Failed to fetch weather data. Please try again later.",
                "status": "error"
            }
            
        current = data.get('current')
        if not current:
            return {
                "error": "Invalid weather data received",
                "status": "error"
            }
            
        return {
            "temperature": current.get('temp_c', 'N/A'),
            "humidity": current.get('humidity', 'N/A'),
            "aqi": current.get('air_quality', {}).get('us-epa-index', 0),
            "condition": current.get('condition', {}).get('text', 'Unknown'),
            "location": data.get('location', {}).get('name', target_city or weather_model.city),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add this with your other router includes (usually where you have other app.include_router() calls)
app.include_router(weather_router, prefix="/api")

# Add this with your other startup event handlers
@app.on_event("startup")
async def startup_event():
    global weather_model
    try:
        API_KEY = os.getenv("WEATHER_API_KEY", "628d4985109c4f6baa3182527250312")
        DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Jammu")
        weather_model = WeatherPredictionModel(API_KEY, DEFAULT_CITY)
        print("✅ Weather service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize weather service: {e}")

    # Seed disease dictionary
    try:
        seed_db = SessionLocal()
        disease_dictionary_tables.seed_disease_dictionary(seed_db)
        seed_db.close()
        print("✅ Disease dictionary seeded")
    except Exception as e:
        print(f"❌ Disease dictionary seeding failed: {e}")

# Make sure this is at the end of the file
# ... (existing imports)
import socketio
from socket_manager import sio_server

# ... (existing code up to app = FastAPI(...))

# Mount the model directory to be served at /api/model
app.mount("/api/model", StaticFiles(directory="../model"), name="model")

# ... (rest of the routes and startup events)

# Wrap FastAPI app with SocketIO
# This makes 'app' variable point to the ASGI app that handles both SocketIO and FastAPI
# Note: We need to use a different variable name for the wrapped app if we want to refer to the FastAPI app later
# But for uvicorn, we want 'app' to be the entry point.
# So we can rename the FastAPI app to 'fastapi_app' or just wrap it at the end.

# Let's keep 'app' as the FastAPI app for clarity in the file, and create 'combined_app' 
# But uvicorn expects 'app'. So we will reassign.

fastapi_app = app
app = socketio.ASGIApp(sio_server, fastapi_app)

if __name__ == "__main__":
    import uvicorn
    # process_workers=1 because TensorFlow might not like multiple processes due to GPU memory
    uvicorn.run(app, host="0.0.0.0", port=8000)
