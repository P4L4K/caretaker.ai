import sys
import os
import cv2
import time
import argparse
import requests
import json
import threading
from datetime import datetime

# Ensure backend directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)
    
# Add BodyMovementDetection to path
bmd_dir = os.path.join(current_dir, "BodyMovementDetection")
if bmd_dir not in sys.path:
    sys.path.append(bmd_dir)

try:
    from VideoMonitoring.united_monitor import UnitedMonitor, draw_united_interface
    from VideoMonitoring.fall_detection import VideoCaptureThread
except ImportError:
    # Try alternate import if running from different location
    try:
        from united_monitor import UnitedMonitor, draw_united_interface
        from fall_detection import VideoCaptureThread
    except ImportError as e:
        print(f"Error importing monitor modules: {e}")
        print("Please make sure you are running this script from the correct directory.")
        sys.exit(1)

# Default configuration
API_URL = "http://127.0.0.1:8000/api"

class LiveMonitorSession:
    def __init__(self, session_id, token=None, camera_index=0, sensitivity="medium", threshold=30):
        self.session_id = session_id
        self.token = token
        self.camera_index = camera_index
        
        # Initialize Monitor
        self.monitor = UnitedMonitor(
            sensitivity=sensitivity, 
            inactivity_threshold=threshold,
            is_live=True,
            process_every_n_frames=2
        )
        
        # State
        self.running = False
        self.last_fall_alert_time = 0
        self.last_inactivity_alert_time = 0
        self.alert_cooldown = 30  # Seconds between alerts
        
    def send_alert(self, alert_type, data):
        """Send alert to backend API"""
        if not self.session_id:
            print(f"[Local Only] {alert_type.upper()} ALERT: {data}")
            return
            
        endpoint = f"{API_URL}/video-monitoring/alert/{self.session_id}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        payload = {
            "alert_type": alert_type,
            "alert_data": data
        }
        
        def _send():
            try:
                print(f"Sending {alert_type} alert to backend...")
                response = requests.post(endpoint, json=payload, headers=headers)
                if response.status_code == 200:
                    print(f"Alert sent successfully!")
                else:
                    print(f"Failed to send alert: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"Error sending alert: {e}")
                
        # Send in background thread to not block video processing
        threading.Thread(target=_send, daemon=True).start()

    def run(self):
        print(f"=== Starting Live Monitor (Session: {self.session_id or 'Local'}) ===")
        print(f"Camera: {self.camera_index}")
        print("Press 'q' to quit")
        
        # Start capture
        cap_thread = VideoCaptureThread(self.camera_index)
        cap_thread.start()
        
        # Wait for camera to warm up
        time.sleep(1.0)
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                # Read frame
                frame, _ = cap_thread.read()
                if frame is None:
                    print("Waiting for frame...")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                results = self.monitor.process_frame(frame)
                
                # Check for alerts
                current_time = time.time()
                global_state = results["global_state"]
                
                # 1. Fall Detection Alert
                if global_state["fall_detected"]:
                    # Check cooldown
                    if current_time - self.last_fall_alert_time > self.alert_cooldown:
                        self.last_fall_alert_time = current_time
                        
                        # Prepare alert data
                        fall_data = {
                            "timestamp": datetime.now().isoformat(),
                            "message": "Fall detected by live monitor",
                            "confidence": "High"
                        }
                        
                        self.send_alert("fall", fall_data)
                
                # 2. Inactivity Alert
                if global_state["inactivity_alert"]:
                    # Check cooldown
                    if current_time - self.last_inactivity_alert_time > self.alert_cooldown:
                        self.last_inactivity_alert_time = current_time
                        
                        inactivity_res = results["inactivity"]
                        alert_data = {
                            "timestamp": datetime.now().isoformat(),
                            "message": f"Inactivity detected ({inactivity_res.get('time_inactive_seconds', 0)}s)",
                            "duration": inactivity_res.get('time_inactive_seconds', 0)
                        }
                        
                        self.send_alert("inactivity", alert_data)
                
                # Visualize
                display_frame = draw_united_interface(frame, results)
                
                if self.session_id:
                     cv2.putText(display_frame, f"Session: {self.session_id[:8]}...", (10, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Live Elderly Monitor", display_frame)
                
                # Handle inputs
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            cap_thread.stop()
            cv2.destroyAllWindows()
            print("Monitor stopped.")

def main():
    parser = argparse.ArgumentParser(description="Live Elderly Monitor Client")
    parser.add_argument("--session-id", type=str, help="Active monitoring session ID")
    parser.add_argument("--token", type=str, help="JWT Authentication token")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--sensitivity", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--threshold", type=int, default=30, help="Inactivity threshold (seconds)")
    
    args = parser.parse_args()
    
    session = LiveMonitorSession(
        session_id=args.session_id,
        token=args.token,
        camera_index=args.camera,
        sensitivity=args.sensitivity,
        threshold=args.threshold
    )
    
    session.run()

if __name__ == "__main__":
    main()
