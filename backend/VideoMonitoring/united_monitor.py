"""
UnitedMonitor — YOLO pose + Bidirectional LSTM fall detection pipeline.

External API is UNCHANGED from the previous version so that stream_manager.py,
video_monitoring routes, and run_live_monitor.py work without modification.

process_frame(frame) → {
    "fall":       {"fall_detected", "timestamp", "keypoints", "boxes",
                   "features", "confidence"},
    "inactivity": {"person_detected", "time_inactive_seconds", "alert",
                   "status", "tracked_box"},
    "global_state": {"fall_detected", "inactivity_alert", "timestamp",
                     "fps", "frame_skip"},
}

draw_united_interface(frame, results, draw_skeleton=True) → annotated frame
"""

import os
import sys
import time
import math

import cv2
import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "BodyMovementDetection"))

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

from fall_lstm_model import FallLSTM, FallSequenceBuffer, load_model, DEFAULT_SEQ_LEN
from inactivity_monitor import InactivityMonitor

# ── Paths ─────────────────────────────────────────────────────────────────────
_YOLO_MODEL_PATH = os.path.join(_here, "..", "yolov8n-pose.pt")
_LSTM_MODEL_PATH = os.path.join(_here, "fall_lstm.pth")

# ── Sensitivity presets ───────────────────────────────────────────────────────
# Live mode uses much stricter thresholds because webcam angles differ from
# training data and the model needs more evidence before committing.
_SENSITIVITY_PRESETS = {
    #                  video mode                                             live mode
    "low":    {"lstm_thresh": 0.92, "confirm_frames": 8,  "confirm_decay": 1, "lstm_thresh_live": 0.95, "confirm_frames_live": 15, "confirm_decay_live": 1},
    "medium": {"lstm_thresh": 0.88, "confirm_frames": 6,  "confirm_decay": 1, "lstm_thresh_live": 0.92, "confirm_frames_live": 12, "confirm_decay_live": 1},
    "high":   {"lstm_thresh": 0.82, "confirm_frames": 4,  "confirm_decay": 1, "lstm_thresh_live": 0.88, "confirm_frames_live": 8,  "confirm_decay_live": 1},
}


# ─────────────────────────────────────────────────────────────────────────────
class UnitedMonitor:
    """
    YOLO pose + LSTM fall detection combined with centroid-based inactivity
    tracking.  Preserves the original __init__ and process_frame API.
    """

    def __init__(
        self,
        sensitivity:           str   = "medium",
        inactivity_threshold:  int   = 30,
        stable_percentage:     float = 0.10,
        is_live:               bool  = False,
        process_every_n_frames: int  = 2,
    ):
        print(f"[UnitedMonitor] YOLO+LSTM pipeline  "
              f"sensitivity={sensitivity}  inactivity={inactivity_threshold}s  "
              f"live={is_live}  skip={process_every_n_frames}")

        self.sensitivity = sensitivity
        self.is_live     = is_live
        self.process_every_n_frames = process_every_n_frames if is_live else 1

        cfg = _SENSITIVITY_PRESETS.get(sensitivity, _SENSITIVITY_PRESETS["medium"])
        if is_live:
            self._lstm_thresh    = cfg["lstm_thresh_live"]
            self._confirm_frames = cfg["confirm_frames_live"]
            self._confirm_decay  = cfg["confirm_decay_live"]
        else:
            self._lstm_thresh    = cfg["lstm_thresh"]
            self._confirm_frames = cfg["confirm_frames"]
            self._confirm_decay  = cfg["confirm_decay"]
        
        # ── Adjust internal gates based on sensitivity ───────────────────────
        mult = 1.0
        if self.sensitivity == "high":    mult = 0.5
        elif self.sensitivity == "low":   mult = 2.0

        # ── State ─────────────────────────────────────────────────────────────
        self._frame_counter        = 0
        self._confirm_count        = 0    # accumulated high-confidence LSTM frames
        self._fall_detected        = False
        self._fall_start_time      = None
        self._inactivity_alert     = False
        self._last_result          = None
 
        # standing-first gate: person must be seen upright before a fall can trigger
        self._has_been_standing      = False
        self._standing_duration      = 0.0   # time (seconds) seen in upright posture
        self._STANDING_TIME_NEEDED   = 0.5 * mult
        self._first_seen_time        = None  # when person first appeared
 
        # Time-based event cooldown: suppresses re-trigger for 10.0 seconds
        self._EVENT_COOLDOWN_SEC     = 10.0 * mult
        self._last_fall_event_time   = -9999.0
        self._fall_event_just_fired  = False
        
        # State-based event gate: ensures only ONE event per continuous fall
        self._is_currently_falling   = False
        self._reset_confirm_frames   = 0   # counter to confirm we've actually stopped falling
        
        # Entry Grace: ignore person for initial bit to prevent teleport/noise
        self._ENTRY_GRACE_SEC        = 0.5 * mult

        # ── YOLO ──────────────────────────────────────────────────────────────
        yolo_path = _YOLO_MODEL_PATH
        if not os.path.isfile(yolo_path):
            yolo_path = os.path.join(_here, "yolov8n-pose.pt")
        self._yolo = YOLO(yolo_path)

        # ── LSTM ──────────────────────────────────────────────────────────────
        self._lstm, self._device = load_model(_LSTM_MODEL_PATH)
        self._seq_buf = FallSequenceBuffer(seq_len=DEFAULT_SEQ_LEN)

        # ── Inactivity monitor (same as before) ───────────────────────────────
        self._inactivity = InactivityMonitor(
            safety_threshold_seconds=inactivity_threshold,
            stability_percentage=stable_percentage,
        )

        # FPS tracking
        self._fps_start   = time.time()
        self._fps_frames  = 0
        self._current_fps = 0.0

    # ── Public API (unchanged) ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Main pipeline entry point.
        """
        self._frame_counter += 1
        now = time.time()
        ts  = now - self._fps_start   # naive relative timestamp

        # Skip logic for live mode performance
        if self.is_live and (self._frame_counter % self.process_every_n_frames != 0):
             if self._last_result:
                 return self._last_result

        # ── YOLO Person + Pose ────────────────────────────────────────────────
        yolo_res = self._yolo.predict(frame, verbose=False, classes=[0], conf=0.3)[0]

        boxes_raw = yolo_res.boxes
        kps_data  = yolo_res.keypoints
        
        boxes    = None
        xy       = None
        conf_kps = None
        
        if boxes_raw is not None and len(boxes_raw.xyxy) > 0:
            boxes = boxes_raw.xyxy.cpu().numpy()   # (P, 4)

        if kps_data is not None and len(kps_data.xy) > 0:
            xy       = kps_data.xy[0].cpu().numpy()    # (17, 2)
            conf_kps = kps_data.conf[0].cpu().numpy()  # (17,)

        # ── LSTM fall detection ────────────────────────────────────────────────
        lstm_prob    = 0.0
        fall_by_lstm = False
        fall_conf    = 0.0
        features     = {}

        if xy is not None:
            # ── Torso Angle Calculation (Upright Check) ────────────────────────
            neck    = (xy[5] + xy[6]) / 2.0
            mid_hip = (xy[11] + xy[12]) / 2.0
            
            # Torso angle relative to vertical
            torso_angle = self._get_angle(neck, mid_hip, vertical=True)
            # Tighten upright angle (30 deg is clearly standing, 50 was too loose)
            is_upright = torso_angle < 30

            # Track upright duration for standing-first gate
            dt = 1.0 / self._current_fps if self._current_fps > 0 else 0.033
            if self._first_seen_time is None:
                self._first_seen_time = now

            if is_upright:
                self._standing_duration += dt
                if self._standing_duration >= self._STANDING_TIME_NEEDED:
                    self._has_been_standing = True
            else:
                # Decay standing duration if not upright
                self._standing_duration = max(0.0, self._standing_duration - (dt * 0.5))

            self._seq_buf.push(xy, conf_kps)

            if self._seq_buf.ready:
                tensor = self._seq_buf.get_tensor(device=self._device)
                with torch.no_grad():
                    logits = self._lstm(tensor)
                    probs  = torch.softmax(logits, dim=1)
                    lstm_prob = float(probs[0, 1])   # fall probability

                fall_by_lstm = lstm_prob >= self._lstm_thresh
                fall_conf    = lstm_prob

            # HUD Metadata
            features = {
                "lstm_prob":           lstm_prob,
                "torso_angle":         torso_angle,
                "is_upright":          is_upright,
                "standing_duration":   round(self._standing_duration, 2)
            }
        else:
            # No person seen
            self._first_seen_time = None

        # ── Geometric gate: bbox aspect ratio ──────────────────────────────────
        bbox_is_horizontal = False
        if boxes is not None and len(boxes) > 0:
            b = boxes[0]
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            if bh > 5.0:
                # Ratio > 0.8 is enough to distinguish from pure standing
                # (Standard human AR is ~0.3-0.5 while standing)
                bbox_is_horizontal = (bw / bh) > 0.8

        # ── Confirmation counter ───────────────────────────────────────────────
        # Logic: We TRUST the LSTM model if:
        # A) It's extremely sure (>98% - High Confidence Bypass)
        # B) OR it's fairly sure + person was previously standing + body is horizontal
        
        high_conf_bypass = (fall_conf > 0.98)
        
        if fall_by_lstm and (high_conf_bypass or (self._has_been_standing and bbox_is_horizontal)):
            self._confirm_count += 1
        else:
            # Decay: subtract self._confirm_decay per non-fall frame (faster reset)
            self._confirm_count = max(0, self._confirm_count - self._confirm_decay)

        current_fall = self._confirm_count >= self._confirm_frames

        # ── Event Trigger Logic (State Machine) ──────────────────────────────
        self._fall_event_just_fired = False
        
        # Use video-time if available (frame_count / fps), else use real time
        effective_now = self._frame_counter / self._current_fps if self._current_fps > 0 else now
        
        # FOR ARCHIVED VIDEOS: Default to a stable 30fps clock for cooldowns
        # unless we've measured a real one. This prevents CPU speed from gaming the system.
        if not self.is_live:
             effective_now = self._frame_counter / 30.0 if self._frame_counter > 0 else 0.0

        time_since_event = effective_now - self._last_fall_event_time
        time_since_entry = now - (self._first_seen_time or now)

        # TRIGGER CONDITIONS:
        # 1. Logic says it's a fall (LSTM + Standing Gate + Geometry Gate)
        # 2. Not currently in a "Fall" state (deduplication)
        # 3. Minimum 10s since last distinct event (safety)
        # 4. Not in entry grace period (prevents "teleport" false positives)
        if (current_fall and 
            not self._is_currently_falling and 
            time_since_event > self._EVENT_COOLDOWN_SEC and
            time_since_entry > self._ENTRY_GRACE_SEC):
            
            self._fall_event_just_fired  = True   # This frame only
            self._is_currently_falling   = True   # Latch until standing
            self._fall_detected          = True
            self._fall_start_time        = now
            self._last_fall_event_time    = effective_now
            
        elif self._is_currently_falling:
            # Requirements to CLEAR the fall state:
            # In LIVE mode: 10s of standing
            # In VIDEO mode: Persistent until gone/reset
            RESET_STANDING_TIME = 10.0 if self.is_live else 600.0 # Extreme latch for videos
            
            if fall_conf < 0.4:
                self._reset_confirm_frames += 1
            else:
                self._reset_confirm_frames = 0
                
            # Disallow reset for videos unless person is gone
            is_clear_of_fall = (self._standing_duration >= RESET_STANDING_TIME)
            
            if is_clear_of_fall or (now - (self._first_seen_time or now) < -2.0):
                self._is_currently_falling = False
                self._fall_detected        = False
                self._confirm_count        = 0
                self._reset_confirm_frames = 0
                self._seq_buf.reset()
                self._has_been_standing    = False 
                self._standing_duration    = 0.0
            else:
                self._fall_detected   = True
                self._fall_start_time = now

        # ── Inactivity ────────────────────────────────────────────────────────
        box_list = []
        if boxes is not None:
            box_list = boxes.tolist()
        inactivity_res = self._inactivity.update(box_list, now)
        self._inactivity_alert = inactivity_res.get("alert", False)

        # ── Build result (same structure as before) ────────────────────────────
        result = {
            "fall": {
                "fall_detected":       self._fall_detected,
                "fall_event_fired":    self._fall_event_just_fired,
                "timestamp":           ts,
                "keypoints":           xy,
                "boxes":               boxes,
                "features":            features,
                "confidence":          round(fall_conf, 4),
                "persons":             1 if xy is not None else 0,
                "bbox_is_horizontal":  bbox_is_horizontal,
            },
            "inactivity": inactivity_res,
            "global_state": {
                "fall_detected":    self._fall_detected,
                "inactivity_alert": self._inactivity_alert,
                "timestamp":        ts,
                "fps":              self._current_fps,
                "frame_skip":       self.process_every_n_frames,
            },
        }

        # Update FPS
        self._fps_frames += 1
        if now - self._fps_start > 1.0:
            self._current_fps = self._fps_frames / (now - self._fps_start)
            # reset window
            self._fps_start  = now
            self._fps_frames = 0

        self._last_result = result
        return result

    def _get_angle(self, p1, p2, vertical=False):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if vertical:
            # Angle relative to (0,1) vector
            angle = math.degrees(math.atan2(abs(dx), abs(dy)))
            return angle
        return math.degrees(math.atan2(dy, dx))

    def reset_fall(self):
        """Call after a fall alert is acknowledged to force fresh evidence."""
        self._fall_detected        = False
        self._fall_start_time      = None
        self._confirm_count        = 0
        self._is_currently_falling = False
        self._has_been_standing    = False
        self._standing_duration    = 0.0
        self._seq_buf.reset()

    @property
    def inactivity_monitor(self):
        return self._inactivity


# ─────────────────────────────────────────────────────────────────────────────
def draw_united_interface(frame: np.ndarray,
                          results: dict,
                          draw_skeleton: bool = True) -> np.ndarray:
    frame_copy   = frame.copy()
    h, w         = frame.shape[:2]
    fall_res     = results["fall"]
    inactivity_r = results["inactivity"]
    gs           = results["global_state"]

    # 1. Skeleton
    keypoints = fall_res.get("keypoints")
    if draw_skeleton and keypoints is not None and len(keypoints) > 0:
        connections = [
            (0,1),(0,2),(1,3),(2,4),
            (5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),
        ]
        line_color = (0, 0, 255) if gs["fall_detected"] else (255, 200, 0)
        for s, e in connections:
            if s < len(keypoints) and e < len(keypoints):
                pt1 = (int(keypoints[s][0]), int(keypoints[s][1]))
                pt2 = (int(keypoints[e][0]), int(keypoints[e][1]))
                cv2.line(frame_copy, pt1, pt2, line_color, 2)

    # 2. Status HUD
    status_y = 40
    color    = (0, 0, 255) if gs["fall_detected"] else (0, 255, 0)
    label    = "FALL DETECTED!" if gs["fall_detected"] else "STATUS: NORMAL"
    cv2.putText(frame_copy, label, (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return frame_copy
