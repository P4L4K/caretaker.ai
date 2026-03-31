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
    #                  video mode               live mode
    "low":    {"lstm_thresh": 0.85, "confirm_frames": 5,  "lstm_thresh_live": 0.95, "confirm_frames_live": 12},
    "medium": {"lstm_thresh": 0.80, "confirm_frames": 4,  "lstm_thresh_live": 0.92, "confirm_frames_live": 10},
    "high":   {"lstm_thresh": 0.72, "confirm_frames": 3,  "lstm_thresh_live": 0.88, "confirm_frames_live": 6},
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

        cfg = _SENSITIVITY_PRESETS.get(sensitivity, _SENSITIVITY_PRESETS["medium"])
        if is_live:
            self._lstm_thresh    = cfg["lstm_thresh_live"]
            self._confirm_frames = cfg["confirm_frames_live"]
        else:
            self._lstm_thresh    = cfg["lstm_thresh"]
            self._confirm_frames = cfg["confirm_frames"]
        print(f"[UnitedMonitor] thresh={self._lstm_thresh}  confirm={self._confirm_frames} frames")

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

        # ── Frame-skip / live settings ────────────────────────────────────────
        self.is_live               = is_live
        self.process_every_n_frames = process_every_n_frames if is_live else 1

        # ── State ─────────────────────────────────────────────────────────────
        self._frame_counter      = 0
        self._confirm_count      = 0    # consecutive high-confidence LSTM frames
        self._fall_detected      = False
        self._fall_start_time    = None
        self._inactivity_alert   = False
        self._last_result        = None

        # FPS tracking
        self._fps_start   = time.time()
        self._fps_frames  = 0
        self._current_fps = 0.0

    # ── Public API (unchanged) ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process one BGR frame.

        Returns dict with keys: "fall", "inactivity", "global_state"
        """
        now = time.time()
        self._frame_counter += 1

        # FPS
        self._fps_frames += 1
        if self._fps_frames % 30 == 0:
            elapsed = now - self._fps_start
            if elapsed > 0:
                self._current_fps = 30.0 / elapsed
            self._fps_start = now

        ts = time.strftime("%H:%M:%S")

        # Frame-skip for live mode
        if self.is_live and (self._frame_counter % self.process_every_n_frames != 0):
            if self._last_result is not None:
                cached = self._last_result.copy()
                cached["global_state"] = {**cached["global_state"], "timestamp": ts}
                return cached

        # ── YOLO inference ─────────────────────────────────────────────────────
        yolo_res = self._yolo(frame, verbose=False)
        kps_data = yolo_res[0].keypoints
        boxes_raw = yolo_res[0].boxes

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
            self._seq_buf.push(xy, conf_kps)

            if self._seq_buf.ready:
                tensor = self._seq_buf.get_tensor(device=self._device)
                with torch.no_grad():
                    logits = self._lstm(tensor)
                    probs  = torch.softmax(logits, dim=1)
                    lstm_prob = float(probs[0, 1])   # fall probability

                fall_by_lstm = lstm_prob >= self._lstm_thresh
                fall_conf    = lstm_prob

            # Expose some features for the HUD (backwards compat)
            features = {
                "lstm_prob":      lstm_prob,
                "torso_angle":    0.0,    # kept for HUD compatibility
                "torso_angle_smooth": 0.0,
                "aspect_ratio":   0.0,
                "aspect_ratio_smooth": 0.0,
                "vertical_speed": 0.0,
            }

        # ── Confirmation counter ───────────────────────────────────────────────
        if fall_by_lstm:
            self._confirm_count += 1
        else:
            self._confirm_count = max(0, self._confirm_count - 1)

        current_fall = self._confirm_count >= self._confirm_frames

        # Persistent fall state (hold for 5 s after last event)
        if current_fall:
            self._fall_detected   = True
            self._fall_start_time = now
        elif self._fall_detected and (now - (self._fall_start_time or 0) > 5.0):
            # Hold period expired — fully reset so next alert needs fresh evidence
            self._fall_detected  = False
            self._confirm_count  = 0
            self._seq_buf.reset()

        # ── Inactivity ────────────────────────────────────────────────────────
        box_list = []
        if boxes is not None:
            box_list = boxes.tolist()
        inactivity_res = self._inactivity.update(box_list, now)
        self._inactivity_alert = inactivity_res.get("alert", False)

        # ── Build result (same structure as before) ────────────────────────────
        result = {
            "fall": {
                "fall_detected": self._fall_detected,
                "timestamp":     ts,
                "keypoints":     xy,
                "boxes":         boxes,
                "features":      features,
                "confidence":    round(fall_conf, 4),
                "persons":       1 if xy is not None else 0,
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

        self._last_result = result
        return result

    def reset_fall(self):
        """Call after a fall alert is acknowledged to force fresh evidence."""
        self._fall_detected   = False
        self._fall_start_time = None
        self._confirm_count   = 0
        self._seq_buf.reset()

    # ── (stream_manager uses this attribute to update threshold dynamically) ───
    @property
    def inactivity_monitor(self):
        """Expose InactivityMonitor so stream_manager can call set_safety_threshold."""
        return self._inactivity


# ─────────────────────────────────────────────────────────────────────────────
def draw_united_interface(frame: np.ndarray,
                          results: dict,
                          draw_skeleton: bool = True) -> np.ndarray:
    """
    Draw combined fall + inactivity HUD on frame.
    Signature and visual layout unchanged from original.
    """
    frame_copy   = frame.copy()
    h, w         = frame.shape[:2]
    fall_res     = results["fall"]
    inactivity_r = results["inactivity"]
    gs           = results["global_state"]

    # ── 1. Skeleton ─────────────────────────────────────────────────────────
    keypoints = fall_res.get("keypoints")
    if draw_skeleton and keypoints is not None and len(keypoints) > 0:
        connections = [
            (0,1),(0,2),(1,3),(2,4),
            (5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),
        ]
        line_color = (0, 0, 255) if gs["fall_detected"] else (255, 100, 0)

        for s, e in connections:
            if s < len(keypoints) and e < len(keypoints):
                x1, y1 = int(keypoints[s][0]), int(keypoints[s][1])
                x2, y2 = int(keypoints[e][0]), int(keypoints[e][1])
                if all(0 <= v for v in [x1, y1, x2, y2]):
                    cv2.line(frame_copy, (x1, y1), (x2, y2), line_color, 2)

        for kp in keypoints:
            x, y_ = int(kp[0]), int(kp[1])
            if 0 <= x < w and 0 <= y_ < h:
                cv2.circle(frame_copy, (x, y_), 3, (0, 255, 255), -1)

    # ── 2. Bounding boxes ────────────────────────────────────────────────────
    boxes       = fall_res.get("boxes", [])
    tracked_box = inactivity_r.get("tracked_box")

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            is_tracked = (tracked_box is not None and
                          len(box) == len(tracked_box) and
                          np.allclose(box, tracked_box))

            if gs["fall_detected"]:
                color, label, thick = (0, 0, 255), "FALL DETECTED", 4
            elif gs["inactivity_alert"] and is_tracked:
                color, label, thick = (0, 165, 255), "INACTIVITY ALERT", 4
            elif is_tracked:
                color, label, thick = (0, 255, 0), "MONITORED", 2
            else:
                color, label, thick = (255, 255, 0), "Visitor", 1

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ── 3. HUD ───────────────────────────────────────────────────────────────
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (0, 0), (320, 220), (0, 0, 0), -1)
    frame_copy = cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0)

    y_pos = 30
    cv2.putText(frame_copy, "CARETAKER.AI MONITOR", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y_pos += 28

    fps        = gs.get("fps", 0)
    frame_skip = gs.get("frame_skip", 1)
    fps_color  = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 10 else (0, 0, 255)
    cv2.putText(frame_copy, f"FPS: {fps:.1f}  skip:1/{frame_skip}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, fps_color, 1)
    y_pos += 24

    # Fall / normal status
    lstm_prob = fall_res.get("features", {}).get("lstm_prob", 0.0)
    if gs["fall_detected"]:
        fstatus, fcolor = "FALL DETECTED!", (0, 0, 255)
    elif not inactivity_r.get("person_detected", False):
        fstatus, fcolor = "Status: No Person", (120, 120, 120)
    else:
        fstatus, fcolor = f"Status: Normal ({lstm_prob:.2f})", (0, 255, 0)

    cv2.putText(frame_copy, fstatus, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fcolor, 2)
    y_pos += 24

    # Inactivity timer
    inactive_sec  = inactivity_r.get("time_inactive_seconds", 0)
    timer_color   = (0, 0, 255) if gs["inactivity_alert"] else (255, 255, 255)
    cv2.putText(frame_copy, f"Inactivity: {inactive_sec}s", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, timer_color, 1)
    y_pos += 24

    status_msg = inactivity_r.get("status", "")
    cv2.putText(frame_copy, status_msg[:32], (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    y_pos += 22

    conf = fall_res.get("confidence", 0.0)
    cv2.putText(frame_copy, f"LSTM confidence: {conf:.3f}", (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)

    # Timestamp top-right
    ts = gs.get("timestamp", "")
    cv2.putText(frame_copy, ts, (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Big central alert overlay
    if gs["fall_detected"]:
        cv2.putText(frame_copy, "FALL DETECTED",
                    (w // 2 - 200, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)
    elif gs["inactivity_alert"]:
        cv2.putText(frame_copy, "INACTIVITY ALERT",
                    (w // 2 - 250, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 165, 255), 3)

    return frame_copy


# ─────────────────────────────────────────────────────────────────────────────
def main():
    """CLI entry-point preserved for backwards compatibility."""
    import argparse
    from fall_detection import VideoCaptureThread

    parser = argparse.ArgumentParser(description="United Fall & Inactivity Monitor")
    parser.add_argument("--video",       type=str, default="")
    parser.add_argument("--camera",      type=int, default=0)
    parser.add_argument("--output",      type=str, default="output.mp4")
    parser.add_argument("--sensitivity", type=str, default="medium",
                        choices=["low", "medium", "high"])
    parser.add_argument("--threshold",   type=int, default=30)
    parser.add_argument("--no-show",     action="store_true")
    parser.add_argument("--frame-skip",  type=int, default=2)
    args = parser.parse_args()

    is_live = not bool(args.video)
    monitor = UnitedMonitor(
        sensitivity=args.sensitivity,
        inactivity_threshold=args.threshold,
        is_live=is_live,
        process_every_n_frames=args.frame_skip if is_live else 1,
    )

    source = args.video if args.video else args.camera
    cap_thread = VideoCaptureThread(source).start()
    frame_width  = int(cap_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_thread.fps

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"Recording to: {args.output}")

    try:
        fc = 0
        while True:
            frame, _ = cap_thread.read()
            if frame is None:
                break
            results      = monitor.process_frame(frame)
            display_frame = draw_united_interface(frame, results)
            if writer:
                writer.write(display_frame)
            if not args.no_show:
                cv2.imshow("United Elderly Monitor", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            fc += 1
            if fc % 100 == 0:
                print(f"Processed {fc} frames…")
    except KeyboardInterrupt:
        pass
    finally:
        cap_thread.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
