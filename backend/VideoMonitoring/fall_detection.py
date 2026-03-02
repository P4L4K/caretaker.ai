"""
High-performance real-time fall detection using YOLOv8 Pose.

Key design goals:
  - Zero-lag camera loop  (capture thread + detector thread run independently)
  - 5-signal fall scoring with per-person state tracking
  - Low false-positive rate via adaptive confirmation window
  - Works on webcam, RTSP streams, and pre-recorded video files
  - Backward-compatible sync API for UnitedMonitor integration
"""

import cv2
import numpy as np
import queue
import threading
import time
import os
import math
import argparse
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ---------------------------------------------------------------------------
# COCO keypoint indices
# ---------------------------------------------------------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _midpoint(kps: np.ndarray, a: int, b: int) -> Tuple[float, float]:
    """Midpoint between two keypoints."""
    return ((kps[a][0] + kps[b][0]) / 2.0, (kps[a][1] + kps[b][1]) / 2.0)


def _angle_from_vertical(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Angle of vector p1->p2 from the vertical axis (0 deg = perfectly upright).
    Uses |atan2(dx, dy)| so 0 = vertical, 90 = horizontal.
    Image coords: Y increases downward, so for a standing person shoulder is
    above hip meaning dy < 0 when going hip->shoulder upward.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 90.0  # degenerate → treat as horizontal
    return abs(math.degrees(math.atan2(abs(dx), abs(dy))))


def _aspect_ratio(kps: np.ndarray, confs: np.ndarray, min_conf: float = 0.25) -> float:
    """Width / Height of the bounding box of visible keypoints."""
    valid = kps[confs > min_conf] if confs is not None else kps[kps[:, 0] > 0]
    if len(valid) < 3:
        return 0.0
    w = valid[:, 0].max() - valid[:, 0].min()
    h = valid[:, 1].max() - valid[:, 1].min()
    return (w / h) if h > 1e-6 else 0.0


def _kp_visible(kps: np.ndarray, confs: np.ndarray, idx: int, min_conf: float = 0.25) -> bool:
    """Check if a keypoint has sufficient confidence."""
    if confs is None:
        return kps[idx][0] > 0 and kps[idx][1] > 0
    return confs[idx] > min_conf


# ---------------------------------------------------------------------------
# Per-person state tracker
# ---------------------------------------------------------------------------

class PersonState:
    """Tracks temporal state for one person across frames."""

    HISTORY = 12           # frames kept for smoothing
    CONFIRM_FRAMES = 3     # default consecutive fall-frames needed
    FAST_CONFIRM = 2       # reduced threshold for high-confidence falls
    FAST_CONF_THR = 0.80   # score above this uses FAST_CONFIRM
    STANDING_ANGLE = 50.0  # torso angle above this → person is upright
    COOLDOWN = 5.0         # seconds before re-alerting same person

    def __init__(self):
        self.angle_hist    = deque(maxlen=self.HISTORY)
        self.aspect_hist   = deque(maxlen=self.HISTORY)
        self.hip_y_hist    = deque(maxlen=self.HISTORY)
        self.sho_y_hist    = deque(maxlen=self.HISTORY)
        self.time_hist     = deque(maxlen=self.HISTORY)
        self.fall_streak   = 0
        self.last_alert    = 0.0
        self.last_upright  = time.time()

    def update(self, angle: float, aspect: float, hip_y: float, sho_y: float, ts: float):
        self.angle_hist.append(angle)
        self.aspect_hist.append(aspect)
        self.hip_y_hist.append(hip_y)
        self.sho_y_hist.append(sho_y)
        self.time_hist.append(ts)
        if angle > self.STANDING_ANGLE:
            self.last_upright = ts

    # --- smoothed accessors ---

    @property
    def smooth_angle(self) -> float:
        return float(np.mean(self.angle_hist)) if self.angle_hist else 90.0

    @property
    def smooth_aspect(self) -> float:
        return float(np.mean(self.aspect_hist)) if self.aspect_hist else 0.0

    @property
    def vertical_speed(self) -> float:
        """Pixels/second downward movement of hip midpoint (positive = falling)."""
        if len(self.hip_y_hist) < 2 or len(self.time_hist) < 2:
            return 0.0
        dt = self.time_hist[-1] - self.time_hist[0]
        if dt < 1e-3:
            return 0.0
        return (self.hip_y_hist[-1] - self.hip_y_hist[0]) / dt

    @property
    def shoulder_drop_speed(self) -> float:
        """Pixels/second downward drop of shoulder midpoint."""
        if len(self.sho_y_hist) < 2 or len(self.time_hist) < 2:
            return 0.0
        dt = self.time_hist[-1] - self.time_hist[0]
        if dt < 1e-3:
            return 0.0
        return (self.sho_y_hist[-1] - self.sho_y_hist[0]) / dt

    def was_standing_recently(self, now: float, window: float = 3.0) -> bool:
        return (now - self.last_upright) < window

    def score_fall(
        self,
        angle_thr: float,
        aspect_thr: float,
        speed_thr: float,
        hip_y: float,
        knee_y: float,
    ) -> Tuple[bool, float]:
        """
        5-signal fall scoring:
          1. Torso angle (0.30)        — torso tilted away from vertical
          2. Aspect ratio (0.25)       — body wider than tall
          3. Hip vertical speed (0.20) — rapid downward motion
          4. Hip below knees (0.15)    — catches side-angle / seated falls
          5. Shoulder drop speed (0.10)— rapid upper-body descent
        Returns (is_fall_frame, confidence 0-1).
        """
        conf = 0.0
        hits = 0

        # Signal 1: Torso angle
        if self.smooth_angle < angle_thr:
            frac = max(0.0, 1.0 - self.smooth_angle / angle_thr)
            conf += 0.30 * frac
            hits += 1

        # Signal 2: Aspect ratio
        if self.smooth_aspect > aspect_thr:
            frac = min(1.0, (self.smooth_aspect - aspect_thr) / max(0.01, aspect_thr))
            conf += 0.25 * frac
            hits += 1

        # Signal 3: Hip vertical speed (downward = positive)
        spd = self.vertical_speed
        if spd > speed_thr:
            frac = min(1.0, (spd - speed_thr) / max(1.0, speed_thr))
            conf += 0.20 * frac
            hits += 1

        # Signal 4: Hip below knees (in image coords, higher Y = lower position)
        hip_below_knees = hip_y > knee_y + 10  # 10px margin
        if hip_below_knees:
            conf += 0.15
            hits += 1

        # Signal 5: Shoulder drop speed
        sho_spd = self.shoulder_drop_speed
        sho_speed_thr = speed_thr * 0.8  # slightly lower threshold for shoulders
        if sho_spd > sho_speed_thr:
            frac = min(1.0, (sho_spd - sho_speed_thr) / max(1.0, sho_speed_thr))
            conf += 0.10 * frac
            hits += 1

        return hits >= 2, min(1.0, conf)


# ---------------------------------------------------------------------------
# Non-blocking video capture thread
# ---------------------------------------------------------------------------

class VideoCaptureThread:
    """Reads frames in background; consumer always gets the latest frame."""

    def __init__(self, source, buffer: int = 4):
        # Handle RTSP URLs and integer camera indices
        if isinstance(source, str) and source.strip() == "":
            source = 0
        elif isinstance(source, str) and source.isdigit():
            source = int(source)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._q     = queue.Queue(maxsize=buffer)
        self._stop  = threading.Event()
        self._t     = threading.Thread(target=self._run, daemon=True)

        self.running = True  # Backward-compat flag for existing code
        self.frame_count = 0

    def start(self):
        self._t.start()
        return self

    def _run(self):
        idx = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._q.put(None)  # EOF signal
                break
            ft = idx / self.fps
            idx += 1
            self.frame_count = idx
            # Drop oldest if full → consumer always sees latest
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put((frame, ft))
        self.cap.release()

    def read(self, timeout: float = 1.0):
        try:
            item = self._q.get(timeout=timeout)
            if item is None:
                return None, None
            return item
        except queue.Empty:
            return None, None

    def stop(self):
        self.running = False
        self._stop.set()
        self._t.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Fall detector
# ---------------------------------------------------------------------------

class _NullCtx:
    """Fallback no-op context manager when torch is not available."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass


class FallDetector:
    """
    Enhanced fall detection with 5-signal scoring and per-person tracking.

    Supports two usage modes:
      1. Sync: call detect_fall(frame) → returns result dict (for UnitedMonitor)
      2. Async: call submit(frame) / result() for zero-stall main loop
    """

    SENSITIVITY_PRESETS = {
        "low":    {"angle": 22.0, "aspect": 1.6, "speed": 60.0},
        "medium": {"angle": 33.0, "aspect": 1.3, "speed": 35.0},
        "high":   {"angle": 45.0, "aspect": 1.1, "speed": 20.0},
    }

    def __init__(
        self,
        *,
        model_name: str = "yolov8n-pose.pt",
        conf: float = 0.30,
        sensitivity: str = "medium",
        imgsz: int = 480,
        cooldown: float = 5.0,
        confirm_frames: int = 3,
        enable_async: bool = False,
    ):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics")

        # --- Resolve model path ---
        if not os.path.exists(model_name):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            for check_dir in [parent_dir, current_dir]:
                candidate = os.path.join(check_dir, model_name)
                if os.path.exists(candidate):
                    model_name = candidate
                    print(f"[FallDetector] Found model at: {model_name}")
                    break

        self.model  = YOLO(model_name)
        self.conf   = conf
        self.imgsz  = imgsz

        # --- Device & precision ---
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            try:
                self.model.model.half()
                print("[FallDetector] GPU + FP16 enabled")
            except Exception:
                print("[FallDetector] GPU enabled (FP32)")
        else:
            self.device = "cpu"
            print("[FallDetector] CPU mode")

        # --- Sensitivity thresholds ---
        p = self.SENSITIVITY_PRESETS.get(sensitivity, self.SENSITIVITY_PRESETS["medium"])
        self.angle_thr   = p["angle"]
        self.aspect_thr  = p["aspect"]
        self.speed_thr   = p["speed"]
        self.cooldown    = cooldown
        self.confirm_frames = confirm_frames

        # --- Per-person state ---
        self._states: Dict[int, PersonState] = {}

        # --- Async threading (optional) ---
        self._async = enable_async
        if enable_async:
            self._in_q:  queue.Queue = queue.Queue(maxsize=2)
            self._out_q: queue.Queue = queue.Queue(maxsize=2)
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._async_loop, daemon=True)
            self._thread.start()

    # ---- Sync API (for UnitedMonitor) ----

    def detect_fall(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame synchronously.
        Returns a dict compatible with UnitedMonitor expectations:
        {
            "fall_detected": bool,
            "timestamp": str,
            "keypoints": ndarray | None,   # first person's keypoints
            "boxes": ndarray | None,
            "features": dict,
            "confidence": float,
            "persons": list,               # all persons with details
        }
        """
        result = self._process(frame)

        # Build backward-compatible keys for UnitedMonitor
        persons = result.get("persons", [])
        first_kps = persons[0]["keypoints"] if persons else None
        first_features = {}
        first_conf = 0.0

        if persons:
            p0 = persons[0]
            first_features = {
                "torso_angle": p0["angle"],
                "torso_angle_smooth": p0["angle"],
                "aspect_ratio": p0["aspect"],
                "aspect_ratio_smooth": p0["aspect"],
                "vertical_speed": p0["speed"],
                "hip_y": p0.get("hip_y", 0),
            }
            first_conf = p0["score"]

        # Collect all boxes
        all_boxes = np.array([p["box"] for p in persons]) if persons else None

        result["keypoints"] = first_kps
        result["boxes"] = all_boxes
        result["features"] = first_features
        result["confidence"] = first_conf
        return result

    # ---- Async API (for standalone / live camera) ----

    def submit(self, frame: np.ndarray):
        """Submit a frame for async processing (non-blocking, drops if busy)."""
        if not self._async:
            return
        if self._in_q.full():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
        self._in_q.put(frame)

    def get_result(self) -> Optional[Dict]:
        """Get latest async result if available (non-blocking)."""
        if not self._async:
            return None
        try:
            return self._out_q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop async processing thread."""
        if self._async:
            self._stop_event.set()
            self._thread.join(timeout=2.0)

    def _async_loop(self):
        while not self._stop_event.is_set():
            try:
                frame = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            result = self._process(frame)
            if self._out_q.full():
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
            self._out_q.put(result)

    # ---- Core processing ----

    def _process(self, frame: np.ndarray) -> Dict:
        now = time.time()
        ts  = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))

        base: Dict[str, Any] = {
            "fall_detected": False,
            "timestamp": ts,
            "persons": [],
        }

        if frame is None or frame.size == 0:
            return base

        try:
            ctx = torch.no_grad() if TORCH_AVAILABLE else _NullCtx()
            with ctx:
                results = self.model.predict(
                    frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    verbose=False,
                    device=self.device,
                )
        except Exception as e:
            print(f"[FallDetector] Inference error: {e}")
            return base

        if not results:
            return base

        r = results[0]
        boxes_raw = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        kps_raw   = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
        confs_raw = (
            r.keypoints.conf.cpu().numpy()
            if r.keypoints is not None and r.keypoints.conf is not None
            else None
        )

        any_fall = False
        persons: List[Dict] = []
        n = len(boxes_raw)

        for i in range(n):
            box  = boxes_raw[i]
            kps  = kps_raw[i]  if kps_raw  is not None and i < len(kps_raw)  else None
            conf = confs_raw[i] if confs_raw is not None and i < len(confs_raw) else None

            if kps is None or len(kps) < 17:
                continue
            if conf is None:
                conf = np.ones(len(kps))

            # --- Extract geometric features with confidence filtering ---
            hip_ok  = _kp_visible(kps, conf, L_HIP) and _kp_visible(kps, conf, R_HIP)
            sho_ok  = _kp_visible(kps, conf, L_SHO) and _kp_visible(kps, conf, R_SHO)
            knee_ok = _kp_visible(kps, conf, L_KNEE) and _kp_visible(kps, conf, R_KNEE)

            if not hip_ok or not sho_ok:
                # Can't compute torso angle without hips and shoulders
                continue

            hip   = _midpoint(kps, L_HIP, R_HIP)
            sho   = _midpoint(kps, L_SHO, R_SHO)
            angle = _angle_from_vertical(hip, sho)
            aspect = _aspect_ratio(kps, conf)
            hip_y = hip[1]
            sho_y = sho[1]

            # Knee Y for hip-below-knee signal
            if knee_ok:
                knee_y = (kps[L_KNEE][1] + kps[R_KNEE][1]) / 2.0
            else:
                knee_y = hip_y + 100  # assume knees are below hips (no data)

            # --- Update per-person state ---
            state = self._states.setdefault(i, PersonState())
            state.update(angle, aspect, hip_y, sho_y, now)

            is_fall_frame, score = state.score_fall(
                self.angle_thr, self.aspect_thr, self.speed_thr,
                hip_y, knee_y,
            )

            # Require person was upright recently (avoids alerting on someone lying still)
            if not state.was_standing_recently(now):
                is_fall_frame = False

            # --- Adaptive confirmation window ---
            if is_fall_frame:
                state.fall_streak += 1
            else:
                state.fall_streak = max(0, state.fall_streak - 1)  # soft decay

            # Use fewer confirmation frames for very high-confidence falls
            required = (
                PersonState.FAST_CONFIRM
                if score >= PersonState.FAST_CONF_THR
                else self.confirm_frames
            )

            confirmed = (
                state.fall_streak >= required
                and (now - state.last_alert) > self.cooldown
            )
            if confirmed:
                state.last_alert  = now
                state.fall_streak = 0
                any_fall = True

            persons.append({
                "id":        i,
                "box":       box,
                "keypoints": kps,
                "confs":     conf,
                "angle":     state.smooth_angle,
                "aspect":    state.smooth_aspect,
                "speed":     state.vertical_speed,
                "score":     score,
                "fall":      confirmed,
                "hip_y":     hip_y,
                "knee_y":    knee_y,
                "sho_speed": state.shoulder_drop_speed,
            })

        base["fall_detected"] = any_fall
        base["persons"]       = persons
        return base


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_detections(frame: np.ndarray, res: Dict[str, Any], fall_detected: bool = None) -> np.ndarray:
    """
    Draw skeletons, boxes, and status overlay.
    Supports both old-style (single-person) and new multi-person result formats.
    """
    out  = frame.copy()
    h, w = out.shape[:2]

    # Determine fall state
    if fall_detected is None:
        fall_detected = res.get("fall_detected", False)

    persons = res.get("persons", [])

    if persons:
        # New multi-person format
        for p in persons:
            is_fall = p.get("fall", False)
            kps  = p.get("keypoints")
            box  = p.get("box")
            color = (0, 0, 255) if is_fall else (0, 220, 0)

            # Bounding box
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 + int(is_fall))

                # Per-person stats
                label = (
                    f"{'FALL! ' if is_fall else ''}"
                    f"A:{p.get('angle', 0):.0f} R:{p.get('aspect', 0):.2f} "
                    f"S:{p.get('speed', 0):.0f}"
                )
                cv2.putText(out, label, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

            # Keypoints
            if kps is not None:
                for j, kp in enumerate(kps):
                    cx, cy = int(kp[0]), int(kp[1])
                    if 0 < cx < w and 0 < cy < h:
                        cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1)

                # Skeleton
                for a, b in SKELETON:
                    if a < len(kps) and b < len(kps):
                        ax, ay = int(kps[a][0]), int(kps[a][1])
                        bx, by = int(kps[b][0]), int(kps[b][1])
                        if 0 < ax < w and 0 < ay < h and 0 < bx < w and 0 < by < h:
                            cv2.line(out, (ax, ay), (bx, by), color, 2)
    else:
        # Backward-compat: old single-person format
        boxes = res.get("boxes")
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if fall_detected else (0, 255, 0)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 + int(fall_detected))

        keypoints = res.get("keypoints")
        if keypoints is not None and len(keypoints) > 0:
            line_color = (0, 0, 255) if fall_detected else (255, 100, 0)
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(out, (x, y), 4, (0, 255, 255), -1)
            for a, b in SKELETON:
                if a < len(keypoints) and b < len(keypoints):
                    x1, y1 = int(keypoints[a][0]), int(keypoints[a][1])
                    x2, y2 = int(keypoints[b][0]), int(keypoints[b][1])
                    if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                        cv2.line(out, (x1, y1), (x2, y2), line_color, 2)

    # --- Status overlay & features ---
    features = res.get("features", {})
    confidence = res.get("confidence", 0.0)

    if fall_detected:
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
        cv2.putText(out, "FALL DETECTED!", (16, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"Confidence: {confidence:.0%}", (16, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "Status: Normal", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2, cv2.LINE_AA)

    # Feature info at bottom
    y_pos = h - 80
    angle = features.get("torso_angle_smooth", features.get("torso_angle", 0))
    aspect = features.get("aspect_ratio_smooth", features.get("aspect_ratio", 0))
    speed = features.get("vertical_speed", 0)
    if angle or aspect or speed:
        cv2.putText(out, f"Angle: {angle:.1f}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(out, f"Aspect: {aspect:.2f}", (20, y_pos + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(out, f"Speed: {speed:.1f}", (20, y_pos + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Timestamp
    cv2.putText(out, res.get("timestamp", ""), (w - 200, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Entry point (standalone mode)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Enhanced Real-time Fall Detection")
    ap.add_argument("--video",       default="",           help="Video file or RTSP URL (omit for webcam)")
    ap.add_argument("--camera",      type=int, default=0,  help="Webcam index")
    ap.add_argument("--output",      default="",           help="Save annotated video to this path")
    ap.add_argument("--sensitivity", default="medium",     choices=["low", "medium", "high"])
    ap.add_argument("--model",       default="yolov8n-pose.pt")
    ap.add_argument("--imgsz",       type=int, default=480,help="Inference size (320=fastest, 640=most accurate)")
    ap.add_argument("--show",        action="store_true",  help="Display video window")
    ap.add_argument("--conf",        type=float, default=0.30)
    args = ap.parse_args()

    source = args.video if args.video else args.camera
    cap    = VideoCaptureThread(source).start()
    det    = FallDetector(
        model_name=args.model,
        conf=args.conf,
        sensitivity=args.sensitivity,
        imgsz=args.imgsz,
        enable_async=True,  # Async mode for standalone
    )

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, cap.fps, (cap.width, cap.height))
        print(f"[INFO] Writing output to {args.output}")

    print(f"[INFO] Starting — sensitivity={args.sensitivity}, imgsz={args.imgsz}, device={det.device}")

    last_result: Dict = {"fall_detected": False, "persons": [], "timestamp": ""}
    frame_n = 0
    t0 = time.time()
    fps_display = 0.0

    try:
        while True:
            item = cap.read(timeout=2.0)
            if item is None or item[0] is None:
                print("[INFO] Stream ended.")
                break
            frame, vid_t = item

            # Submit to async detector
            det.submit(frame)

            # Grab latest result if available
            r = det.get_result()
            if r is not None:
                last_result = r
                if r["fall_detected"]:
                    for fp in r["persons"]:
                        if fp["fall"]:
                            print(f"\n{'='*55}")
                            print(f"  FALL DETECTED  |  {r['timestamp']}")
                            print(f"  Angle  : {fp['angle']:.1f} deg")
                            print(f"  Aspect : {fp['aspect']:.2f}")
                            print(f"  Speed  : {fp['speed']:.0f} px/s")
                            print(f"  Score  : {fp['score']:.2f}")
                            print(f"{'='*55}\n")

            vis = draw_detections(frame, last_result)

            # FPS counter
            frame_n += 1
            if frame_n % 20 == 0:
                fps_display = 20.0 / max(1e-3, time.time() - t0)
                t0 = time.time()
            cv2.putText(vis, f"FPS {fps_display:.1f}", (cap.width - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

            if args.show:
                cv2.imshow("Fall Detection  [Q=quit]", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break

            if writer:
                writer.write(vis)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cap.stop()
        det.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()