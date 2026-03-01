"""
AudioDetectionService — mirrors the logic from coughandsneezedetection/app.py.

Primary engine: Acoustic Feature Analyzer (rule-based, always active).
Secondary engine: CRNN model (optional, boosts confidence when loaded).

The acoustic analyzer is the reliable detection engine.
The CRNN model is used ONLY to confirm/boost a detection that the
acoustic analyzer already flagged — it does NOT override it.
"""

import os
import numpy as np
import logging
import datetime
from typing import Dict, Any, Optional

try:
    from utils.audio_model.architecture import build_model
    from utils.audio_model.preprocessing import preprocess_chunk, SAMPLE_RATE
    from utils.audio_model.acoustic_analyzer import classify_acoustic
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from utils.audio_model.architecture import build_model
    from utils.audio_model.preprocessing import preprocess_chunk, SAMPLE_RATE
    from utils.audio_model.acoustic_analyzer import classify_acoustic

logger = logging.getLogger(__name__)

# Match original app.py constants
CONFIDENCE_THRESHOLD = 0.60   # 60%
CHUNK_DURATION       = 0.5    # 500 ms
CHUNK_SAMPLES        = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_RATIO        = 0.5
SLIDE_SAMPLES        = int(CHUNK_SAMPLES * (1 - OVERLAP_RATIO))

CLASS_LABELS = ["Cough", "Sneeze", "Talking", "Noise"]


class AudioDetectionService:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        # Sliding window buffer — same as original app.py
        self._buffer = np.array([], dtype=np.float32)
        self._load_model()

    def _load_model(self):
        """Load CRNN weights if available. Mirrors original app.py startup."""
        print("=" * 52)
        print("  SonicGuard CRNN -- Loading ...")
        print("=" * 52)

        try:
            self.model = build_model()
            base_dir = os.path.dirname(os.path.dirname(__file__))
            weights_path = os.path.join(base_dir, "utils", "audio_model", "weights.weights.h5")

            if os.path.isfile(weights_path):
                try:
                    self.model.load_weights(weights_path)
                    self.model_loaded = True
                    print(f"  [OK] Loaded weights from {weights_path}")
                except Exception as e:
                    print(f"  [WARN] Could not load weights: {e}")
            else:
                print("  [INFO] No weights found -- using acoustic analyzer only.")

            # Warm-up
            _dummy = np.zeros((1, 64, 101, 1), dtype=np.float32)
            self.model(_dummy, training=False)
            print("  [OK] Model warmed up")

        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            self.model_loaded = False

        print("  [OK] Acoustic Feature Analyzer active -- detection ready!")

    def reset_buffer(self):
        """Reset the sliding window buffer (call on client connect/disconnect)."""
        self._buffer = np.array([], dtype=np.float32)

    def process_audio_chunk(self, audio_bytes: bytes):
        """
        Append incoming PCM bytes to the sliding window buffer and
        yield classification results for each complete window.

        This exactly mirrors the while-loop in the original handle_audio_chunk().

        Returns a list of result dicts (may be empty if buffer not full yet).
        """
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.float32)
        except Exception:
            return []

        self._buffer = np.concatenate([self._buffer, samples])
        results = []

        while len(self._buffer) >= CHUNK_SAMPLES:
            window = self._buffer[:CHUNK_SAMPLES]

            # ── Acoustic Feature Analysis (primary, always runs) ────
            result = classify_acoustic(window, sr=SAMPLE_RATE)

            # ── CRNN boost (secondary, only when model is loaded) ───
            # Only boost if acoustic already suspects Cough/Sneeze
            if self.model_loaded and result["predicted"] in ("Cough", "Sneeze"):
                try:
                    input_tensor = preprocess_chunk(window, sr=SAMPLE_RATE)
                    prediction = self.model(input_tensor, training=False)
                    probs = prediction.numpy()[0]

                    dl_scores = {
                        "Cough":   float(probs[0] * 100),
                        "Sneeze":  float(probs[1] * 100),
                        "Talking": float(probs[2] * 100),
                        "Noise":   float(probs[3] * 100),
                    }

                    # Blend: 70% acoustic (reliable) + 30% DL (texture)
                    for cls in CLASS_LABELS:
                        result[cls] = round(result[cls] * 0.70 + dl_scores[cls] * 0.30, 2)

                    # Recompute winner after blend
                    predicted = max(CLASS_LABELS, key=lambda c: result[c])
                    result["predicted"]  = predicted
                    result["confidence"] = result[predicted]

                except Exception as e:
                    logger.error(f"DL inference failed: {e}")

            result["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
            results.append(result)

            # Slide forward
            self._buffer = self._buffer[SLIDE_SAMPLES:]

        return results


# Global singleton — loaded once at startup
audio_service = AudioDetectionService()
