import cv2
import numpy as np
import time
import argparse
import sys
import os
import threading

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "BodyMovementDetection"))

from fall_detection import FallDetector, VideoCaptureThread, draw_detections, SKELETON
from inactivity_monitor import InactivityMonitor

class UnitedMonitor:
    def __init__(self, sensitivity="medium", inactivity_threshold=30, stable_percentage=0.10, 
                 is_live=False, process_every_n_frames=2):
        print(f"[Init] United Monitor: Sensitivity={sensitivity}, InactivityThreshold={inactivity_threshold}s")
        print(f"[Init] Live Mode: {is_live}, Frame Skip: {process_every_n_frames}")
        
        # Initialize sub-monitors
        self.fall_detector = FallDetector(sensitivity=sensitivity)
        self.inactivity_monitor = InactivityMonitor(
            safety_threshold_seconds=inactivity_threshold,
            stability_percentage=stable_percentage
        )
        
        # State
        self.fall_detected = False
        self.fall_start_time = None
        self.inactivity_alert = False
        
        # Performance optimization for live feeds
        self.is_live = is_live
        self.process_every_n_frames = process_every_n_frames if is_live else 1
        self.frame_counter = 0
        self.last_result = None  # Cache last result for skipped frames
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

    def process_frame(self, frame):
        """
        Process a single frame through both Fall Detection and Inactivity Monitoring.
        Implements frame skipping for live feeds to improve performance.
        """
        current_time = time.time()
        self.frame_counter += 1
        
        # Frame skipping logic for live feeds
        should_process = (self.frame_counter % self.process_every_n_frames == 0)
        
        # Always process if fall was recently detected (high priority)
        if self.fall_detected and (current_time - self.fall_start_time < 5.0):
            should_process = True
        
        # Skip frame processing if not needed
        if not should_process and self.last_result is not None:
            # Update timestamp in cached result
            cached_result = self.last_result.copy()
            cached_result["global_state"]["timestamp"] = time.strftime("%H:%M:%S")
            return cached_result
        
        # 1. Run Fall Detector (uses enhanced 5-signal scoring)
        # Returns: {fall_detected, boxes, keypoints, features, confidence, timestamp, persons}
        fall_res = self.fall_detector.detect_fall(frame)
        
        # Update persistent fall state
        if fall_res.get("fall_detected"):
            self.fall_detected = True
            self.fall_start_time = current_time
        
        # Reset fall state after 5 seconds if no new fall is detected
        if self.fall_detected and (current_time - self.fall_start_time > 5.0):
            self.fall_detected = False
        
        # 2. Run Inactivity Monitor
        # Reuse boxes from Fall Detector (YOLO)
        boxes = fall_res.get("boxes")
        if boxes is None:
            boxes = []
        elif isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()
        
        # InactivityMonitor.update takes list of boxes and current time
        inactivity_res = self.inactivity_monitor.update(boxes, current_time)
        
        # Update persistent inactivity alert state
        self.inactivity_alert = inactivity_res.get("alert", False)
        
        # Calculate FPS
        self.fps_frame_count += 1
        if self.fps_frame_count % 30 == 0:
            elapsed = current_time - self.fps_start_time
            if elapsed > 0:
                self.current_fps = 30 / elapsed
            self.fps_start_time = current_time
        
        result = {
            "fall": fall_res,
            "inactivity": inactivity_res,
            "global_state": {
                "fall_detected": self.fall_detected,
                "inactivity_alert": self.inactivity_alert,
                "timestamp": fall_res.get("timestamp"),
                "fps": self.current_fps,
                "frame_skip": self.process_every_n_frames
            }
        }
        
        # Cache result for frame skipping
        self.last_result = result
        return result

def draw_united_interface(frame, results, draw_skeleton=True):
    """
    Draw a combined interface for both Fall and Inactivity monitoring.
    Args:
        frame: The image frame
        results: The detection results
        draw_skeleton: improving user experience by hiding skeleton lines if needed (default: True)
    """
    frame_copy = frame.copy()
    h, w = frame.shape[:2]
    
    fall_res = results["fall"]
    inactivity_res = results["inactivity"]
    global_state = results["global_state"]
    
    # --- 1. Draw Skeletons (from Fall Detection) ---
    keypoints = fall_res.get("keypoints")
    if draw_skeleton and keypoints is not None and len(keypoints) > 0:
        # Draw skeleton lines
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        # Color: Red if falling, else Blue-ish
        line_color = (0, 0, 255) if global_state["fall_detected"] else (255, 100, 0)
        
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(frame_copy, (x1, y1), (x2, y2), line_color, 2)
                    
        # Draw joints
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame_copy, (x, y), 3, (0, 255, 255), -1)

    # --- 2. Draw Bounding Boxes & Status (Combined Logic) ---
    boxes = fall_res.get("boxes", [])
    tracked_box = inactivity_res.get("tracked_box")
    
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Identify if this is the tracked person for inactivity
            is_tracked = False
            if tracked_box is not None:
                # Simple coordinate match check or reference check
                # Since InactivityMonitor returns the exact box object/array from input, 'is' might work or exact equality
                if np.array_equal(box, tracked_box):
                    is_tracked = True

            # Determine Box Color & Label
            if global_state["fall_detected"]:
                color = (0, 0, 255) # Red for Fall
                label = "FALL DETECTED"
                thickness = 4
            elif global_state["inactivity_alert"] and is_tracked:
                color = (0, 165, 255) # Orange for Inactivity Alert
                label = "INACTIVITY ALERT"
                thickness = 4
            elif is_tracked:
                color = (0, 255, 0) # Green for Normal Tracked
                label = "MONITORED"
                thickness = 2
            else:
                color = (255, 255, 0) # Cyan/Yellow for others
                label = "Visitor"
                thickness = 1
                
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- 3. Heads-Up Display (HUD) ---
    
    # Background for HUD
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (0, 0), (320, 210), (0, 0, 0), -1)
    alpha = 0.6
    frame_copy = cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0)
    
    y_pos = 30
    
    # Title
    cv2.putText(frame_copy, "UNITED DETECTOR", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    
    # FPS Display (Performance Indicator)
    fps = global_state.get("fps", 0)
    frame_skip = global_state.get("frame_skip", 1)
    fps_color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 10 else (0, 0, 255)
    cv2.putText(frame_copy, f"FPS: {fps:.1f} | Skip: 1/{frame_skip}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
    y_pos += 25
    
    # Fall Status
    person_detected = inactivity_res.get("person_detected", False)
    
    if global_state["fall_detected"]:
        fall_status = "FALL DETECTED!" 
        fall_color = (0, 0, 255)
    elif not person_detected:
        fall_status = "Status: No Person"
        fall_color = (100, 100, 100) # Grey
    else:
        fall_status = "Status: Normal"
        fall_color = (0, 255, 0)
        
    cv2.putText(frame_copy, fall_status, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fall_color, 2)
    y_pos += 25
    
    # Inactivity Timer
    inactive_time = inactivity_res.get("time_inactive_seconds", 0)
    alert_status = inactivity_res.get("alert", False)
    timer_color = (0, 0, 255) if alert_status else (255, 255, 255)
    cv2.putText(frame_copy, f"Inactivity Timer: {inactive_time}s", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, timer_color, 1)
    y_pos += 25

    # Detailed Status
    status_msg = inactivity_res.get("status", "")
    cv2.putText(frame_copy, status_msg[:30], (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_pos += 25
    
    # Fall Metrics (Angle/Aspect/Speed from 5-signal scoring)
    features = fall_res.get("features", {})
    angle = features.get("torso_angle_smooth", features.get("torso_angle", 0))
    aspect = features.get("aspect_ratio_smooth", features.get("aspect_ratio", 0))
    speed = features.get("vertical_speed", 0)
    conf = fall_res.get("confidence", 0)
    cv2.putText(frame_copy, f"A:{angle:.0f} R:{aspect:.2f} S:{speed:.0f} C:{conf:.0%}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)

    # Timestamp (Top Right)
    ts = global_state.get("timestamp", "")
    cv2.putText(frame_copy, ts, (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Big Central Alert Overlay
    if global_state["fall_detected"]:
        cv2.putText(frame_copy, "FALL DETECTED", (w//2 - 200, h//2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)
    elif global_state["inactivity_alert"]:
        cv2.putText(frame_copy, "INACTIVITY ALERT", (w//2 - 250, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 165, 255), 3)

    return frame_copy

def main():
    parser = argparse.ArgumentParser(description="United Fall & Inactivity Monitor")
    parser.add_argument("--video", type=str, default="", help="Path to input video file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video")
    parser.add_argument("--sensitivity", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--threshold", type=int, default=30, help="Inactivity threshold in seconds")
    parser.add_argument("--no-show", action="store_true", help="Do not display window")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every N frames (for live camera)")
    args = parser.parse_args()

    # Determine if live mode (camera vs video file)
    is_live = not bool(args.video)
    
    # Initialize Monitor with performance settings
    monitor = UnitedMonitor(
        sensitivity=args.sensitivity, 
        inactivity_threshold=args.threshold,
        is_live=is_live,
        process_every_n_frames=args.frame_skip if is_live else 1
    )

    # Initialize Source
    source = args.video if args.video else args.camera
    print(f"Starting capture from: {source}")
    cap_thread = VideoCaptureThread(source).start()
    
    # Get dimensions
    frame_width = int(cap_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_thread.fps
    
    # Initialize Writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"Recording to: {args.output}")

    try:
        frame_count = 0
        while True:
            frame, _ = cap_thread.read()
            if frame is None:
                print("End of stream.")
                break
                
            # Process
            results = monitor.process_frame(frame)
            
            # Visualize
            display_frame = draw_united_interface(frame, results)
            
            # Write
            if writer:
                writer.write(display_frame)
                
            # Show
            if not args.no_show:
                cv2.imshow("United Elderly Monitor", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User Quit.")
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap_thread.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
