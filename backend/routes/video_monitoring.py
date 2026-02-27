from fastapi import APIRouter, Depends, HTTPException, status, Header, UploadFile, File, BackgroundTasks, Request, Form
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from typing import Optional, List, Dict
from pydantic import BaseModel
import cv2
import numpy as np
import json
import os
import sys
import uuid
import shutil
import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import backend modules
sys.path.append(str(Path(__file__).parent.parent))

from config import get_db, SessionLocal
from repository.users import JWTRepo, UsersRepo
from tables.users import CareTaker, CareRecipient
from tables.video_analysis import VideoAnalysis
from utils.email import send_fall_alert_email
# Import FastMail for direct email sending with attachments if needed
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from dotenv import load_dotenv

# Import stream manager and united monitor
sys.path.append(str(Path(__file__).parent.parent / "VideoMonitoring"))
from stream_manager import stream_manager
from united_monitor import UnitedMonitor, draw_united_interface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=['Video Monitoring'], prefix='/video-monitoring')

# ============================================================================
# PERSISTENCE HANDLER
# ============================================================================
STATUS_FILE = Path("processing_status.json")

def load_process_status():
    """Load processing status from JSON file"""
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load status file: {e}")
    return {}

def save_process_status(status_dict):
    """Save processing status to JSON file"""
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(status_dict, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save status file: {e}")

# Load status on startup
process_status = load_process_status()

# Global inactivity threshold (can be updated per user)
inactivity_thresholds = {}  # username -> threshold_seconds

# Models
class StartLiveRequest(BaseModel):
    camera_index: int = 0
    sensitivity: str = "medium"

class InactivityThresholdRequest(BaseModel):
    threshold_seconds: int

class UpdateSessionThresholdRequest(BaseModel):
    threshold_seconds: int


def _get_username_from_auth(auth_header: Optional[str]):
    """Extract username from JWT token"""
    if not auth_header:
        return None
    
    try:
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
            
        token = parts[1]
        decoded = JWTRepo.decode_token(token)
        
        if not decoded or not isinstance(decoded, dict):
            return None
            
        return decoded.get('sub')
        
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        return None


# ============================================================================
# EMAIL HELPER WITH ATTACHMENT SUPPORT
# ============================================================================
load_dotenv()
conf = ConnectionConfig(
    MAIL_USERNAME = os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD"),
    MAIL_FROM = os.getenv("MAIL_FROM"),
    MAIL_PORT = int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER = os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS = True,
    MAIL_SSL_TLS = False,
    USE_CREDENTIALS = True
)

async def send_alert_email_with_image(recipient_email: str, subject: str, body: str, image_bytes: Optional[bytes] = None, image_name: str = "alert.jpg"):
    """Send email with optional image attachment"""
    try:
        # Build message kwargs
        message_kwargs = {
            "subject": subject,
            "recipients": [recipient_email],
            "body": body,
            "subtype": "html"
        }
        
        # Only add attachments if we have image bytes
        if image_bytes:
            message_kwargs["attachments"] = [(image_name, image_bytes, "image/jpeg")]
        
        message = MessageSchema(**message_kwargs)
        fm = FastMail(conf)
        await fm.send_message(message)
        logger.info(f"Alert email sent to {recipient_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


# ============================================================================
# UPLOAD AND VIDEO FILE ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/upload-video")
async def upload_video_for_analysis(
    file: UploadFile = File(...),
    recipient_id: Optional[int] = Form(None),
    background_tasks: BackgroundTasks = None,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Upload a video file for fall detection and inactivity analysis.
    Returns a process_id to track the analysis status.
    """
    # Authenticate user
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    try:
        # Validate file
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Save uploaded file
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / unique_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Prepare output directory
        output_dir = Path("processed_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"analyzed_{unique_filename}"
        output_path = output_dir / output_filename
        
        # Start video processing in background
        process_id = str(uuid.uuid4())
        
        # Initialize process status
        process_status[process_id] = {
            "status": "processing",
            "progress": 0,
            "user_email": user.email,
            "username": username,
            "caretaker_id": user.id,
            "recipient_id": recipient_id,
            "input_file": str(file_path),
            "output_file": str(output_path),
            "output_filename": output_filename,
            "start_time": time.time(),
            "error": None,
            "falls_detected": [],
            "has_falls": False
        }
        save_process_status(process_status)
        
        # Run processing in background
        # Usage of background_tasks.add_task with a regular def runs in thread pool
        if background_tasks:
            background_tasks.add_task(
                process_video_direct,
                process_id=process_id,
                input_path=str(file_path),
                output_path=str(output_path),
                user_email=user.email,
                caretaker_id=user.id,
                recipient_id=recipient_id
            )
        
        return {
            "status": "success",
            "process_id": process_id,
            "message": "Video uploaded successfully. Processing started."
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def process_video_direct(process_id: str, input_path: str, output_path: str, user_email: str, caretaker_id: int, recipient_id: Optional[int] = None):
    """
    Direct in-process video analysis using UnitedMonitor.
    Runs in a thread pool managed by FastAPI BackgroundTasks.
    """
    try:
        logger.info(f"Starting direct analysis for {process_id}")
        
        # Initialize Monitor
        monitor = UnitedMonitor(
            sensitivity="medium",
            inactivity_threshold=30,
            is_live=False,
            process_every_n_frames=1
        )
        
        # Open Video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Could not open input video")
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize Writer with H.264 codec for browser compatibility
        # Try H.264 first (best browser support), fallback to mp4v if not available
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise Exception("H.264 codec not available")
            logger.info(f"Using H.264 (avc1) codec for video encoding")
        except:
            logger.warning("H.264 codec not available, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        falls_detected = []
        last_alert_time = 0
        
        # Create window for showcase
        window_name = "Video Analysis - Processing"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process Frame
            results = monitor.process_frame(frame)
            
            # Draw Interface
            display_frame = draw_united_interface(frame, results)
            
            # Write Frame
            writer.write(display_frame)
            
            # Display for showcase (every 2nd frame to reduce lag)
            if frame_count % 2 == 0:
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(1)  # 1ms delay for window update
            
            # Track Alerts
            if results["global_state"]["fall_detected"]:
                current_time = time.time()
                # Simple debounce (5 seconds)
                if current_time - last_alert_time > 5.0:
                    timestamp = results["global_state"].get("timestamp", datetime.now().isoformat())
                    falls_detected.append({
                        "timestamp": timestamp,
                        "message": "Fall Detected",
                        "frame": frame_count
                    })
                    last_alert_time = current_time
            
            frame_count += 1
            
            # Update progress every 30 frames
            if frame_count % 30 == 0:
                progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                if process_id in process_status:
                    process_status[process_id]["progress"] = progress
                    # Optimization: Don't save to disk on every frame, maybe every 10%
                    if progress % 10 == 0:
                        save_process_status(process_status)
        
        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyWindow(window_name)
        
        # Final Status Update
        if process_id in process_status:
            process_status[process_id].update({
                "status": "completed",
                "progress": 100,
                "falls_detected": falls_detected,
                "has_falls": len(falls_detected) > 0,
                "output_filename": os.path.basename(output_path), 
                "completed_at": datetime.now().isoformat()
            })
            save_process_status(process_status)
            
            # Send Email if falls detected
            if len(falls_detected) > 0:
                asyncio.run(send_alert_email_with_image(
                    recipient_email=user_email,
                    subject="⚠️ Fall Detected in Uploaded Video",
                    body=f"Analysis detected {len(falls_detected)} fall events. Please review the processed video on your dashboard."
                    # No image attachment for now
                ))

            # --- SAVE TO DATABASE ---
            try:
                db_session = SessionLocal()
                
                # Mock metrics for now (since we don't have detailed tracking yet)
                fall_count = len(falls_detected)
                activity_score = max(0.0, 10.0 - (fall_count * 2))  # Reduce by 2 for each fall
                mobility_score = max(0.0, 10.0 - (fall_count * 1.5))
                inactivity_secs = 0 # Placeholder
                
                analysis_entry = VideoAnalysis(
                    recipient_id=recipient_id,
                    caretaker_id=caretaker_id,
                    video_filename=os.path.basename(output_path),
                    has_fall=len(falls_detected) > 0,
                    fall_count=fall_count,
                    inactivity_duration_seconds=inactivity_secs,
                    activity_score=activity_score,
                    mobility_score=mobility_score,
                    timestamp=datetime.utcnow()
                )
                db_session.add(analysis_entry)
                db_session.commit()
                logger.info(f"Saved analysis to DB for Recipient {recipient_id}")
            except Exception as e:
                logger.error(f"Failed to save analysis to DB: {e}")
            finally:
                db_session.close()
        
        # Remove input file
        if os.path.exists(input_path):
            os.remove(input_path)
            
        logger.info(f"Analysis completed for {process_id}")

    except Exception as e:
        logger.error(f"Analysis error for {process_id}: {e}", exc_info=True)
        if process_id in process_status:
            process_status[process_id].update({
                "status": "error",
                "error": str(e)
            })
            save_process_status(process_status)
        # Try cleanup
        if os.path.exists(input_path):
             os.remove(input_path)
        # Close window if it exists
        try:
            cv2.destroyAllWindows()
        except:
            pass


@router.get("/status/{process_id}")
async def get_processing_status(
    process_id: str,
    authorization: Optional[str] = Header(None)
):
    """
    Get the status of a video processing job.
    """
    global process_status  # Declare at the start before any usage
    
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    if process_id not in process_status:
        # Reload from disk to be sure
        process_status = load_process_status()
        
    if process_id not in process_status:
        raise HTTPException(status_code=404, detail="Process not found")
    
    state = process_status[process_id].copy()
    
    # Add elapsed time
    if "start_time" in state:
        elapsed = time.time() - state["start_time"]
        state["elapsed_seconds"] = round(elapsed, 2)
    
    # Frontend will construct the absolute URL from output_filename
    # No need to add processed_url here
        
    return state


@router.get("/history")
async def get_monitoring_history(
    authorization: Optional[str] = Header(None)
):
    """
    Get history of processed videos.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    try:
        output_dir = Path("processed_videos")
        if not output_dir.exists():
            return []
            
        files = []
        for file in output_dir.glob("analyzed_*.mp4"):
            stats = file.stat()
            files.append({
                "filename": file.name,
                "timestamp": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "size": stats.st_size
                # Frontend will construct absolute URL from filename
            })
            
        # Sort by timestamp descending
        files.sort(key=lambda x: x["timestamp"], reverse=True)
        return files
        
    except Exception as e:
        logger.error(f"Error listing history: {e}")
        return []


@router.get("/stats")
async def get_monitoring_stats(
    authorization: Optional[str] = Header(None)
):
    """
    Get statistics for video monitoring.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
        
    try:
        output_dir = Path("processed_videos")
        total_uploads = len(list(output_dir.glob("analyzed_*.mp4"))) if output_dir.exists() else 0
        
        total_alerts = 0
        for session_id in stream_manager.get_active_sessions():
            s = stream_manager.get_session(session_id)
            if s:
                total_alerts += len(s.get_alerts())
            
        return {
            "total_uploads": total_uploads,
            "total_alerts": total_alerts
        }
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {
            "total_uploads": 0,
            "total_alerts": 0
        }


# ============================================================================
# NEW STREAMING ENDPOINTS FOR WEB-BASED VIDEO MONITORING
# ============================================================================

@router.post("/start")
async def start_live_monitoring(
    request: StartLiveRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Start a new live monitoring session with MJPEG streaming.
    Returns session_id and stream_url for embedding in dashboard.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    user = UsersRepo.find_by_username(db, CareTaker, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    # Check for existing active session for this user
    for session_id in stream_manager.get_active_sessions():
        session = stream_manager.get_session(session_id)
        if session and session.username == username:
             return {
                "status": "already_active",
                "session_id": session_id,
                "stream_url": f"/api/video-monitoring/stream/{session_id}",
                "message": "You already have an active session"
            }
    
    # Create new session
    session_id = str(uuid.uuid4())
    inactivity_threshold = inactivity_thresholds.get(username, 30)
    
    success = stream_manager.create_session(
        session_id=session_id,
        camera_index=request.camera_index,
        sensitivity=request.sensitivity,
        inactivity_threshold=inactivity_threshold
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start camera session. Camera might be busy.")
    
    # Assign ownership
    session = stream_manager.get_session(session_id)
    if session:
        session.username = username
        session.user_email = user.email
    
    return {
        "status": "success",
        "session_id": session_id,
        "stream_url": f"/api/video-monitoring/stream/{session_id}",
        "message": "Live monitoring started successfully"
    }


# IMPORTANT: More specific routes must come BEFORE generic routes
# /session/{session_id}/alerts must be before /stream/{session_id}
# Otherwise FastAPI will match "session" as a session_id value in /stream/{session_id}

@router.get("/session/{session_id}/alerts")
async def get_session_alerts(
    session_id: str,
    since: Optional[str] = None,
    authorization: Optional[str] = Header(None)
):
    """
    Get alerts for an active session.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    alerts = session.get_alerts(since=since)
    
    return {
        "session_id": session_id,
        "alerts": alerts,
        "total": len(alerts)
    }


@router.get("/stream/{session_id}")
async def stream_video(session_id: str, request: Request):
    """
    MJPEG stream endpoint for live video feed.
    Handles client disconnects to stop the stream generator.
    """
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    logger.info(f"Stream requested for session {session_id}, running={session.running}")
    
    async def generate():
        """Generator function for MJPEG stream with disconnect check"""
        frame_counter = 0
        try:
            while session.running:
                # Check for client disconnect
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from stream {session_id}")
                    break

                frame_bytes = session.get_frame()
                if frame_bytes:
                    frame_counter += 1
                    if frame_counter % 30 == 0:  # Log every 30 frames
                        logger.debug(f"Stream {session_id}: Sent {frame_counter} frames, size={len(frame_bytes)} bytes")
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # FPS limiting - target ~25 FPS
                    await asyncio.sleep(0.04)
                else:
                    # No frame available, wait a bit
                    if frame_counter == 0:
                        logger.warning(f"Stream {session_id}: No frames available yet")
                    await asyncio.sleep(0.05)
                    
        except asyncio.CancelledError:
            logger.info(f"Stream {session_id} cancelled")
        except Exception as e:
            logger.error(f"Stream {session_id} error: {e}", exc_info=True)
        finally:
            logger.info(f"Stream {session_id} ended after {frame_counter} frames")
            
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/stop/{session_id}")
async def stop_live_monitoring(
    session_id: str,
    authorization: Optional[str] = Header(None)
):
    """
    Stop an active monitoring session and release camera.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    session = stream_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.username != username:
        raise HTTPException(status_code=403, detail="Not authorized to stop this session")
    
    stream_manager.stop_session(session_id)
    return {"status": "success", "message": "Monitoring session stopped"}


@router.post("/update-threshold/{session_id}")
async def update_session_threshold(
    session_id: str,
    request: UpdateSessionThresholdRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Update inactivity threshold for a running session.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    session = stream_manager.get_session(session_id)
    if not session:
         raise HTTPException(status_code=404, detail="Session not found")

    if session.username != username:
         raise HTTPException(status_code=403, detail="Not authorized")
         
    if request.threshold_seconds < 5:
        raise HTTPException(status_code=400, detail="Threshold too low")
        
    session.update_threshold(request.threshold_seconds)
    
    return {"status": "success", "message": f"Threshold updated to {request.threshold_seconds}s"}


@router.post("/set-inactivity-threshold")
async def set_inactivity_threshold(
    request: InactivityThresholdRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Set default inactivity threshold for user.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    if request.threshold_seconds < 5 or request.threshold_seconds > 300:
        raise HTTPException(status_code=400, detail="Threshold must be between 5 and 300 seconds")
    
    inactivity_thresholds[username] = request.threshold_seconds
    
    return {
        "status": "success",
        "threshold_seconds": request.threshold_seconds,
        "message": f"Inactivity threshold set to {request.threshold_seconds} seconds"
    }

@router.get("/get-inactivity-threshold")
async def get_inactivity_threshold(authorization: Optional[str] = Header(None)):
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    threshold = inactivity_thresholds.get(username, 30)
    return {"threshold_seconds": threshold}


@router.get("/recipient/{recipient_id}/stats")
async def get_recipient_video_stats(
    recipient_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    Get aggregated video analysis stats for a specific recipient.
    """
    username = _get_username_from_auth(authorization)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
    
    # 1. Get all analysis records for this recipient
    analyses = db.query(VideoAnalysis).filter(VideoAnalysis.recipient_id == recipient_id).all()
    
    if not analyses:
        return {
            "total_videos": 0,
            "total_falls": 0,
            "avg_activity_score": 0,
            "avg_mobility_score": 0,
            "history": []
        }
    
    # 2. Aggregate Data
    total_videos = len(analyses)
    total_falls = sum(r.fall_count for r in analyses)
    avg_activity = sum(r.activity_score for r in analyses) / total_videos if total_videos > 0 else 0
    avg_mobility = sum(r.mobility_score for r in analyses) / total_videos if total_videos > 0 else 0
    
    # 3. Format History for Graphs (last 10 sessions)
    history = []
    # Sort chronologically
    sorted_analyses = sorted(analyses, key=lambda x: x.timestamp)
    for r in sorted_analyses:
        history.append({
            "date": r.timestamp.isoformat(),
            "falls": r.fall_count,
            "activity": r.activity_score,
            "mobility": r.mobility_score
        })
        
    return {
        "total_videos": total_videos,
        "total_falls": total_falls,
        "avg_activity_score": round(avg_activity, 1),
        "avg_mobility_score": round(avg_mobility, 1),
        "history": history
    }
