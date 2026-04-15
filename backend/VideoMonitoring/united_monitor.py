"""
UnitedMonitor — YOLO pose + Bidirectional LSTM fall detection pipeline.

External API is UNCHANGED:
process_frame(frame) → {
    "fall":       {"fall_detected", "all_persons", "keypoints", "boxes", "features", ...},
    "inactivity": {"person_detected", "time_inactive_seconds", "alert"},
    "global_state": {"fall_detected", "inactivity_alert", ...}
}
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "BodyMovementDetection"))

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

from fall_lstm_model import FallLSTM, FallSequenceBuffer, load_model, DEFAULT_SEQ_LEN
from inactivity_monitor import InactivityMonitor

_YOLO_MODEL_PATH = os.path.join(_here, "..", "yolov8n-pose.pt")
_LSTM_MODEL_PATH = os.path.join(_here, "fall_lstm.pth")

_SENSITIVITY_PRESETS = {
    "low":    {"lstm_thresh": 0.88, "confirm_frames": 8, "confirm_decay": 2},
    "medium": {"lstm_thresh": 0.85, "confirm_frames": 6, "confirm_decay": 2},
    "high":   {"lstm_thresh": 0.80, "confirm_frames": 4, "confirm_decay": 2},
}

class UnitedMonitor:
    def __init__(self, sensitivity="medium", inactivity_threshold=30, stable_percentage=0.10, is_live=False, process_every_n_frames=2):
        print(f"[UnitedMonitor] Initializing... live={is_live}")
        self.sensitivity = sensitivity
        self.is_live = is_live
        self.process_every_n_frames = process_every_n_frames if is_live else 1
        
        cfg = _SENSITIVITY_PRESETS.get(sensitivity, _SENSITIVITY_PRESETS["medium"])
        self._lstm_thresh = cfg["lstm_thresh"]
        self._confirm_frames = cfg["confirm_frames"]
        self._confirm_decay = cfg["confirm_decay"]

        # ── Adjust internal gates based on sensitivity ───────────────────────
        mult = 1.0
        if self.sensitivity == "high":    mult = 0.5
        elif self.sensitivity == "low":   mult = 2.0

        self._frame_counter = 0
        self._confirm_count = 0
        self._fall_detected = False
        self._fall_start_time = None
        self._inactivity_alert = False
        self._last_result = None
        
        # standing-first gate
        self._has_been_standing = False
        self._standing_duration = 0.0
        self._STANDING_TIME_NEEDED = 0.5 * mult

        # State-based event gate
        self._is_currently_falling = False
        self._reset_confirm_frames = 0
        self._last_fall_event_time = -9999.0
        self._fall_event_just_fired = False

        # Alert Suppression / Cooldown
        self._EVENT_COOLDOWN_SEC = 10.0 * mult
        self._ENTRY_GRACE_SEC = 0.5 * mult
        self._last_event_frame = -999

        # Tracking state — once the monitored_id is locked it never auto-switches
        self._monitored_id = None
        self._monitored_id_locked = False   # True after first person is chosen
        self._monitored_centroid = None

        # FPS tracking
        self._fps_start = time.time()
        self._fps_frames = 0
        self._current_fps = 0.0

        self._yolo = YOLO(_YOLO_MODEL_PATH)
        self._lstm, self._device = load_model(_LSTM_MODEL_PATH)
        self._seq_buf = FallSequenceBuffer(seq_len=DEFAULT_SEQ_LEN)
        self._inactivity = InactivityMonitor(safety_threshold_seconds=inactivity_threshold, stability_percentage=stable_percentage)


    def process_frame(self, frame: np.ndarray) -> dict:
        self._frame_counter += 1
        self._fall_event_just_fired = False  # Reset per frame
        now = time.time()
        ts = now - self._fps_start

        if self.is_live and (self._frame_counter % self.process_every_n_frames != 0):
            if self._last_result: return self._last_result

        # YOLO Tracking (Stable IDs)
        yolo_res = self._yolo.track(frame, persist=True, verbose=False, classes=[0], conf=0.3)[0]
        boxes_raw = yolo_res.boxes
        kps_data = yolo_res.keypoints
        
        all_persons = []
        possible_humans = []
        monitored_idx = -1
        
        if boxes_raw is not None and len(boxes_raw.xyxy) > 0:
            boxes = boxes_raw.xyxy.cpu().numpy()
            # YOLO IDs (may be None if not tracking)
            yolo_ids = boxes_raw.id.cpu().numpy().astype(int) if boxes_raw.id is not None else None
            
            for i in range(len(boxes)):
                if kps_data is not None and len(kps_data.xy) > i:
                    kp_xy = kps_data.xy[i].cpu().numpy()
                    kp_conf = kps_data.conf[i].cpu().numpy()
                    y_id = yolo_ids[i] if yolo_ids is not None else i
                    
                    box = boxes[i]
                    area = (box[2]-box[0]) * (box[3]-box[1])
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    
                    possible_humans.append({
                        "id": y_id, "box": box, "xy": kp_xy, "conf": kp_conf, 
                        "cx": cx, "cy": cy, "area": area
                    })

            if possible_humans:
                # ── Sticky ID Tracking (Finalized Logic) ───────────────────
                # Priority: MONITORED_ID -> Centroid -> Largest Subject
                target = None

                if self._monitored_id is not None:
                    matches = [p for p in possible_humans if p["id"] == self._monitored_id]
                    if matches:
                        target = matches[0]

                if target is None and self._monitored_centroid is not None:
                    best_dist = 1e6
                    for p in possible_humans:
                        dist = math.sqrt((p["cx"] - self._monitored_centroid[0])**2 + (p["cy"] - self._monitored_centroid[1])**2)
                        if dist < best_dist and dist < 250:
                            best_dist = dist
                            target = p

                if target is None:
                    # Choose largest subject as the primary monitored person
                    target = max(possible_humans, key=lambda p: p["area"])
                    self._monitored_id_locked = True

                if target is not None:
                    monitored_idx = target["id"]
                    self._monitored_id = monitored_idx
                    self._monitored_centroid = (target["cx"], target["cy"])
                    self._monitored_was_present = True
                    self._monitored_last_seen = now
                else:
                    monitored_idx = -1

                for p in possible_humans:
                    is_mon = (p["id"] == monitored_idx)
                    all_persons.append({
                        "box": p["box"], "xy": p["xy"], "is_monitored": is_mon,
                        "label": "MONITORED" if is_mon else "VISITOR"
                    })
            else:
                # No humans at all — keep the ID but clear centroid
                monitored_idx = -1
                self._monitored_centroid = None
                self._monitored_was_present = False
        else:
            self._monitored_id = None
            self._monitored_centroid = None

        # Fall Detection on Monitored Person
        lstm_prob = 0.0
        features = {}
        monitored_xy = None
        monitored_box = None
        bbox_is_horizontal = False
        dt = 1.0 / self._current_fps if self._current_fps > 0 else 0.033

        if monitored_idx != -1:
            p = [p for p in possible_humans if p["id"] == monitored_idx][0]
            monitored_xy, monitored_box, conf_kps = p["xy"], p["box"], p["conf"]

            neck = (monitored_xy[5] + monitored_xy[6]) / 2.0
            mid_hip = (monitored_xy[11] + monitored_xy[12]) / 2.0
            torso_angle = self._get_angle(neck, mid_hip, vertical=True)
            is_upright = torso_angle < 35

            # Compute bbox aspect ratio — horizontal bbox = person is lying down
            bw = monitored_box[2] - monitored_box[0]
            bh = monitored_box[3] - monitored_box[1]
            bbox_is_horizontal = (bw > bh * 0.8) if bh > 0 else False

            if is_upright:
                self._standing_duration += dt
                if self._standing_duration > 0.5: self._has_been_standing = True
            else:
                self._standing_duration = max(0.0, self._standing_duration - dt)

            self._seq_buf.push(monitored_xy, conf_kps)
            if self._seq_buf.ready:
                with torch.no_grad():
                    logits = self._lstm(self._seq_buf.get_tensor(device=self._device))
                    lstm_prob = float(torch.softmax(logits, dim=1)[0, 1])

            features = {"lstm_prob": lstm_prob, "is_upright": is_upright, "standing_duration": self._standing_duration}
            
            # Confirmation counter (Finalized 0.8 Geometric Gate)
            high_conf_bypass = (lstm_prob > 0.98)
            
            if (lstm_prob >= self._lstm_thresh) and (high_conf_bypass or (self._has_been_standing and bbox_is_horizontal)):
                self._confirm_count += 1
            else:
                self._confirm_count = max(0, self._confirm_count - self._confirm_decay)

            # Trigger logic from finalized version
            effective_now = self._frame_counter / self._current_fps if self._current_fps > 0 else now
            if not self.is_live:
                 effective_now = self._frame_counter / 30.0

            time_since_event = effective_now - self._last_fall_event_time
            
            if (self._confirm_count >= self._confirm_frames and 
                not self._is_currently_falling and 
                time_since_event > self._EVENT_COOLDOWN_SEC):
                self._fall_event_just_fired = True
                self._is_currently_falling = True
                self._fall_detected = True
                self._last_fall_event_time = effective_now
                self._last_event_frame = self._frame_counter
            
            # HUD Clearing Logic: once clearly upright and stable, clear the fall flag
            if self._is_currently_falling:
                if is_upright and self._standing_duration > 2.0:
                    self._fall_detected = False
                    if self._standing_duration > 10.0:
                        self._is_currently_falling = False
                        self._confirm_count = 0


        # Inactivity
        inactivity_res = self._inactivity.update([monitored_box.tolist()] if monitored_box is not None else [], now)
        self._inactivity_alert = inactivity_res.get("alert", False)

        result = {
            "fall": {
                "fall_detected": self._fall_detected,
                "fall_event_fired": self._fall_event_just_fired,
                "all_persons": all_persons,
                "features": features,
                "keypoints": monitored_xy,
                "boxes": monitored_box
            },
            "inactivity": inactivity_res,
            "global_state": {
                "fall_detected": self._fall_detected,
                "inactivity_alert": self._inactivity_alert,
                "fps": self._current_fps,
                "timestamp": ts
            }
        }
        
        self._fps_frames += 1
        if now - self._fps_start > 1.0:
            self._current_fps = self._fps_frames / (now - self._fps_start)
            self._fps_start, self._fps_frames = now, 0

        self._last_result = result
        return result

    def _get_angle(self, p1, p2, vertical=False):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        return math.degrees(math.atan2(abs(dx), abs(dy))) if vertical else math.degrees(math.atan2(dy, dx))

    def reset_fall(self):
        self._fall_detected = False
        self._is_currently_falling = False
        self._confirm_count = 0
        self._seq_buf.reset()

def draw_united_interface(frame, results, draw_skeleton=True):
    frame_copy = frame.copy()
    fall_res = results["fall"]
    gs = results["global_state"]
    all_persons = fall_res.get("all_persons", [])
    h, w = frame_copy.shape[:2]

    # ── Draw bounding boxes and labels for each detected person ──
    for p in all_persons:
        box, is_mon = p["box"], p["is_monitored"]
        # Use plain ASCII text — OpenCV cannot render Unicode/emoji
        if is_mon:
            if gs["fall_detected"]:
                color = (0, 0, 255)       # Red — fall
                label = "! FALL DETECTED"
            elif gs["inactivity_alert"]:
                color = (0, 128, 255)     # Orange — inactivity
                label = ">> INACTIVITY ALERT"
            else:
                color = (0, 200, 0)       # Green — monitored OK
                label = ">> MONITORED"
        else:
            color = (0, 200, 255)         # Yellow — visitor
            label = "VISITOR"

        x1, y1, x2, y2 = map(int, box)
        thickness = 3 if is_mon else 2
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)

        # Draw label with background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame_copy, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_copy, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if draw_skeleton and is_mon and p["xy"] is not None:
            kps = p["xy"]
            conn = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
            for s, e in conn:
                if s < len(kps) and e < len(kps):
                    cv2.line(frame_copy, (int(kps[s][0]), int(kps[s][1])), (int(kps[e][0]), int(kps[e][1])), color, 2)

    # ── Status bar at top of frame ── always visible even with no detections ──
    bar_h = 30
    cv2.rectangle(frame_copy, (0, 0), (w, bar_h), (20, 20, 20), -1)

    fps_val = gs.get("fps", 0.0)
    n_persons = len(all_persons)
    monitored_found = any(p["is_monitored"] for p in all_persons)

    # Build status string
    if n_persons == 0:
        status_txt = "Scanning... (step back so full body is visible)"
        status_color = (100, 100, 100)
    elif monitored_found:
        if gs["fall_detected"]:
            status_txt = "FALL DETECTED"
            status_color = (0, 0, 255)
        elif gs["inactivity_alert"]:
            status_txt = "INACTIVITY ALERT"
            status_color = (0, 128, 255)
        else:
            status_txt = "Monitoring active"
            status_color = (0, 200, 0)
    else:
        status_txt = f"Visitors in frame: {n_persons}"
        status_color = (0, 200, 255)

    fps_txt = f"FPS:{fps_val:.1f}"
    cv2.putText(frame_copy, status_txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 2)
    cv2.putText(frame_copy, fps_txt, (w - 80, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return frame_copy

