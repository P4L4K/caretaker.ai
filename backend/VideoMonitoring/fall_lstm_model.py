"""
fall_lstm_model.py — LSTM model architecture for fall detection.

Components:
  - FallLSTM      : Bidirectional GRU + self-attention + classifier
  - FallSequenceBuffer : Rolling 30-frame buffer for inference
  - normalize_keypoints : Hip-centred, torso-scaled normalisation
  - load_model    : Load a saved checkpoint
"""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_SEQ_LEN  = 30    # frames per sequence
INPUT_FEATURES   = 34   # 17 COCO keypoints × 2 (x, y)
HIDDEN_SIZE      = 128
NUM_LAYERS       = 2
DROPOUT          = 0.4

# COCO keypoint indices used for normalisation
L_HIP,  R_HIP  = 11, 12
L_SHO,  R_SHO  =  5,  6
# ─────────────────────────────────────────────────────────────────────────────


def normalize_keypoints(xy: np.ndarray,
                        conf: np.ndarray | None = None,
                        min_conf: float = 0.2) -> np.ndarray:
    """
    Normalize 17 COCO keypoints to be hip-centred and torso-scaled.

    Args:
        xy   : (17, 2) float array of keypoint coordinates (pixels)
        conf : (17,)  float array of keypoint confidences (optional)
        min_conf : confidences below this are zeroed out

    Returns:
        (34,) float32 normalised feature vector
    """
    kps = xy.copy().astype(np.float32)

    # Zero out low-confidence keypoints
    if conf is not None:
        for i in range(len(kps)):
            if conf[i] < min_conf:
                kps[i] = [0.0, 0.0]

    hip_mid = (kps[L_HIP] + kps[R_HIP]) / 2.0
    sho_mid = (kps[L_SHO] + kps[R_SHO]) / 2.0
    torso   = float(np.linalg.norm(sho_mid - hip_mid))

    if torso < 1e-4:
        return np.zeros(INPUT_FEATURES, dtype=np.float32)

    norm = (kps - hip_mid) / torso

    # Re-zero points that were originally zero (missing)
    for i in range(len(kps)):
        if xy[i, 0] == 0.0 and xy[i, 1] == 0.0:
            norm[i] = [0.0, 0.0]

    return norm.flatten().astype(np.float32)


# ── Attention layer ──────────────────────────────────────────────────────────

class _SelfAttention(nn.Module):
    """Additive self-attention over the time dimension."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)   # ×2 because BiGRU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, hidden*2)
        scores  = self.attn(x).squeeze(-1)          # (batch, seq)
        weights = torch.softmax(scores, dim=-1)     # (batch, seq)
        context = (weights.unsqueeze(-1) * x).sum(dim=1)  # (batch, hidden*2)
        return context


# ── Main Model ───────────────────────────────────────────────────────────────

class FallLSTM(nn.Module):
    """
    Bidirectional GRU with self-attention for fall/normal classification.

    Input  : (batch, seq_len, 34)
    Output : (batch, 2)   logits — 0=normal, 1=fall
    """

    def __init__(self,
                 input_size:  int = INPUT_FEATURES,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers:  int = NUM_LAYERS,
                 dropout:     float = DROPOUT):
        super().__init__()

        self.bn_input = nn.BatchNorm1d(input_size)

        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        self.attention = _SelfAttention(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        batch, seq, feat = x.shape

        # BatchNorm expects (batch*seq, features)
        x = x.reshape(batch * seq, feat)
        x = self.bn_input(x)
        x = x.reshape(batch, seq, feat)

        out, _ = self.gru(x)          # (batch, seq, hidden*2)
        context = self.attention(out) # (batch, hidden*2)
        logits  = self.classifier(context)
        return logits


# ── Sequence buffer for inference ────────────────────────────────────────────

class FallSequenceBuffer:
    """
    Rolling ring buffer that accumulates 30-frame normalised keypoint vectors.
    Use during inference (video or live) to feed the LSTM.

    Usage:
        buf = FallSequenceBuffer(seq_len=30)
        for frame in video:
            buf.push(xy, conf)
            if buf.ready:
                tensor = buf.get_tensor(device)
                logits = model(tensor)
    """

    def __init__(self, seq_len: int = DEFAULT_SEQ_LEN):
        self.seq_len = seq_len
        self._buf: deque = deque(maxlen=seq_len)

    @property
    def ready(self) -> bool:
        return len(self._buf) == self.seq_len

    def push(self, xy: np.ndarray, conf: np.ndarray | None = None):
        """Push one frame's keypoints into the buffer."""
        vec = normalize_keypoints(xy, conf)
        self._buf.append(vec)

    def get_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Return a (1, seq_len, 34) float32 tensor ready for the model."""
        arr = np.stack(list(self._buf), axis=0)   # (seq_len, 34)
        t   = torch.from_numpy(arr).unsqueeze(0)  # (1, seq_len, 34)
        return t.to(device)

    def reset(self):
        self._buf.clear()


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def load_model(path: str,
               input_size:  int = INPUT_FEATURES,
               hidden_size: int = HIDDEN_SIZE,
               num_layers:  int = NUM_LAYERS) -> tuple[FallLSTM, str]:
    """
    Load a FallLSTM checkpoint.

    Returns:
        (model, device_str)

    Raises:
        FileNotFoundError if path does not exist — run train_lstm.py first.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[FallLSTM] Model not found: {path}\n"
            "  → Run extract_keypoints.py first, then train_lstm.py."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = FallLSTM(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers)

    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        val_acc = checkpoint.get("val_accuracy", "?")
        val_f1  = checkpoint.get("val_f1", "?")
        print(f"[FallLSTM] Loaded checkpoint: acc={val_acc}  f1={val_f1}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, device
