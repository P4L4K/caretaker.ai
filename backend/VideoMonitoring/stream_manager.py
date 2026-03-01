"""
MJPEG Stream Manager for Live Video Monitoring
Handles camera sessions and frame streaming for web dashboard
"""
import cv2
import numpy as np
import threading
import time
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import queue
import logging

from united_monitor import UnitedMonitor, draw_united_interface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraSession:
    """Manages a single camera monitoring session with MJPEG streaming"""
    
    def __init__(self, session_id: str, camera_index: int = 0, 
                 sensitivity: str = "medium", inactivity_threshold: int = 30):
        self.session_id = session_id
        self.camera_index = camera_index
        self.sensitivity = sensitivity
        self.inactivity_threshold = inactivity_threshold
        
        # User Ownership
        self.username: Optional[str] = None
        self.user_email: Optional[str] = None
        
        # Threading
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Frame buffer
        self.current_frame: Optional[np.ndarray] = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        
        # Alerts
        self.last_alert_frame: Optional[bytes] = None
        
        # Camera and monitor
        self.cap: Optional[cv2.VideoCapture] = None
        self.monitor: Optional[UnitedMonitor] = None
        
        # Session info
        self.start_time = datetime.now()
        self.last_activity = datetime.now()
        self.alerts: List[dict] = []
        self.frame_count = 0
        
    def start(self) -> bool:
        """Start the camera session"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera index {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize unified monitor
            self.monitor = UnitedMonitor(
                sensitivity=self.sensitivity,
                inactivity_threshold=self.inactivity_threshold,
                is_live=True,
                process_every_n_frames=2  # Process every 2nd frame for performance
            )
            
            # Start capture thread
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Session {self.session_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera session: {e}", exc_info=True)
            return False
    
    def update_threshold(self, seconds: int):
        """Update inactivity threshold dynamically"""
        with self.lock:
            self.inactivity_threshold = seconds
            if self.monitor and hasattr(self.monitor, 'inactivity_monitor'):
                # Use the setter method on InactivityMonitor
                if hasattr(self.monitor.inactivity_monitor, 'set_safety_threshold'):
                    self.monitor.inactivity_monitor.set_safety_threshold(seconds)
                    logger.info(f"Session {self.session_id}: Inactivity threshold updated to {seconds}s")
                else:
                    # Fallback: attribute is named safety_threshold, not safety_threshold_seconds
                    self.monitor.inactivity_monitor.safety_threshold = seconds
                    logger.info(f"Session {self.session_id}: Inactivity threshold updated (direct) to {seconds}s")

    def _capture_loop(self):
        """Main capture and processing loop"""
        while self.running:
            start_time = time.time()
            try:
                if self.cap is None:
                    break

                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Check monitor existence
                if not self.monitor:
                    continue

                # Process frame through unified monitor
                results = self.monitor.process_frame(frame)
                
                # Draw interface (Hide skeletons for live feed as requested)
                display_frame = draw_united_interface(frame, results, draw_skeleton=False)
                
                # Check for alerts
                alert_triggered = False
                if results["global_state"]["fall_detected"]:
                    self._add_alert("fall", "Fall detected!")
                    alert_triggered = True
                elif results["global_state"]["inactivity_alert"]:
                    self._add_alert("inactivity", "Inactivity alert!")
                    alert_triggered = True
                
                # Update current frame
                with self.lock:
                    self.current_frame = display_frame.copy()
                    self.frame_count += 1
                    self.last_activity = datetime.now()
                    
                    if alert_triggered:
                        # Save alert frame as bytes
                        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            self.last_alert_frame = buffer.tobytes()
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(display_frame)
                except queue.Full:
                    # Skip frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(display_frame)
                    except:
                        pass
                
                # FPS Control: Target ~25 FPS (0.04s per frame)
                elapsed = time.time() - start_time
                sleep_time = max(0.04 - elapsed, 0.001)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def _add_alert(self, alert_type: str, message: str):
        """Add an alert to the session"""
        # Simple debounce: check if last alert of same type was recent (< 5s)
        now = datetime.now()
        recent = [a for a in self.alerts if a["type"] == alert_type and 
                  (now - datetime.fromisoformat(a["timestamp"])) < timedelta(seconds=5)]
        
        if not recent:
            alert = {
                "type": alert_type,
                "message": message,
                "timestamp": now.isoformat(),
                "frame_number": self.frame_count
            }
            self.alerts.append(alert)
            logger.info(f"[Alert] Session {self.session_id} - {alert_type}: {message}")
    
    def get_frame(self) -> Optional[bytes]:
        """Get current frame as JPEG bytes"""
        with self.lock:
            if self.current_frame is None:
                logger.debug(f"Session {self.session_id}: No frame available (current_frame is None)")
                return None
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', self.current_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                if self.frame_count % 100 == 0:  # Log every 100 frames
                    logger.debug(f"Session {self.session_id}: Frame {self.frame_count}, size={len(frame_bytes)} bytes")
                return frame_bytes
            else:
                logger.warning(f"Session {self.session_id}: Failed to encode frame")
                return None
    
    def get_alerts(self, since: Optional[str] = None) -> list:
        """Get alerts, optionally filtered by timestamp"""
        if since:
            return [a for a in self.alerts if a["timestamp"] > since]
        return self.alerts.copy()
    
    def stop(self):
        """Stop the camera session"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        logger.info(f"Session {self.session_id} stopped")


class StreamManager:
    """Manages multiple camera sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, CameraSession] = {}
        self.lock = threading.Lock()
    
    def create_session(self, session_id: str, camera_index: int = 0,
                      sensitivity: str = "medium", 
                      inactivity_threshold: int = 30) -> bool:
        """Create and start a new camera session"""
        
        # Auto-cleanup before creating new session
        self.cleanup_inactive_sessions()
        
        with self.lock:
            if session_id in self.sessions:
                return False
            
            session = CameraSession(
                session_id=session_id,
                camera_index=camera_index,
                sensitivity=sensitivity,
                inactivity_threshold=inactivity_threshold
            )
            
            if session.start():
                self.sessions[session_id] = session
                return True
            return False
    
    def get_session(self, session_id: str) -> Optional[CameraSession]:
        """Get a session by ID"""
        with self.lock:
            return self.sessions.get(session_id)
    
    def stop_session(self, session_id: str) -> bool:
        """Stop and remove a session"""
        with self.lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.stop()
                return True
            return False
    
    def get_active_sessions(self) -> list:
        """Get list of active session IDs"""
        with self.lock:
            return list(self.sessions.keys())
    
    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours or explicitly stopped/broken"""
        with self.lock:
            now = datetime.now()
            to_remove = []
            
            for sid, session in self.sessions.items():
                age = (now - session.start_time).total_seconds() / 3600
                
                # Check for thread health
                if session.thread and not session.thread.is_alive():
                    logger.warning(f"Session {sid} thread died unexpectedly. Cleaning up.")
                    to_remove.append(sid)
                    continue

                if age > max_age_hours:
                    logger.info(f"Session {sid} expired (age > {max_age_hours}h). Cleaning up.")
                    to_remove.append(sid)
            
            for sid in to_remove:
                session = self.sessions.pop(sid, None)
                if session:
                    session.stop()

# Global stream manager instance
stream_manager = StreamManager()
