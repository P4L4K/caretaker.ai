"""
STEP 2: Train FallLSTM on extracted keypoint sequences.

Run AFTER extract_keypoints.py has completed.
Saves best checkpoint to: backend/VideoMonitoring/fall_lstm.pth

Training strategy:
  - GroupShuffleSplit (80/20) splits by event group → zero data leakage
  - Weighted CrossEntropyLoss handles class imbalance (more falls than normals)
  - BiGRU + attention model from fall_lstm_model.py
  - Cosine annealing LR + gradient clipping
"""

import os
import sys
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.model_selection import GroupShuffleSplit
except ImportError:
    print("[ERROR] scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from fall_lstm_model import FallLSTM

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(_here, "data")
MODEL_OUT = os.path.join(_here, "fall_lstm.pth")

EPOCHS    = 50
BATCH     = 32
LR        = 1e-3
VAL_RATIO = 0.20
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── Load data ────────────────────────────────────────────────────────────
    for fname in ("sequences.npy", "labels.npy", "groups.npy"):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(path):
            print(f"[ERROR] Missing: {path}")
            print("  → Run extract_keypoints.py first.")
            sys.exit(1)

    X = np.load(os.path.join(DATA_DIR, "sequences.npy"))                  # (N,30,34)
    y = np.load(os.path.join(DATA_DIR, "labels.npy"))                     # (N,)
    g = np.load(os.path.join(DATA_DIR, "groups.npy"), allow_pickle=True)  # (N,)

    print(f"\n[Data] sequences : {X.shape}")
    print(f"[Data] fall      : {int(y.sum())}  normal : {int((y == 0).sum())}")
    print(f"[Data] event groups : {len(set(g))} unique\n")

    # ── Group-aware split ─────────────────────────────────────────────────────
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=g))

    # Assert zero leakage
    train_groups = set(g[train_idx])
    val_groups   = set(g[val_idx])
    overlap      = train_groups & val_groups
    assert len(overlap) == 0, f"[FATAL] Group leakage detected: {overlap}"
    print(f"[Split] Train: {len(train_idx)} seqs across {len(train_groups)} groups")
    print(f"[Split] Val  : {len(val_idx)}   seqs across {len(val_groups)} groups")
    print("[Split] ✓ No group leakage confirmed\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ── Weighted loss for class imbalance ─────────────────────────────────────
    counts  = np.bincount(y)   # [normal_count, fall_count]
    total   = counts.sum()
    weights = torch.tensor(
        [total / (2.0 * counts[0]), total / (2.0 * counts[1])],
        dtype=torch.float32
    ).to(device)
    print(f"[Loss] Class weights → normal={weights[0]:.3f}  fall={weights[1]:.3f}")
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ── Data loaders ──────────────────────────────────────────────────────────
    def make_loader(idx, shuffle: bool) -> DataLoader:
        Xt = torch.from_numpy(X[idx].astype(np.float32))
        yt = torch.from_numpy(y[idx])
        return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model     = FallLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1  = 0.0
    best_acc = 0.0

    print(f"\n{'Epoch':>5}  {'Loss':>8}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        tp = fp = fn = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds   = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += len(yb)
                tp += ((preds == 1) & (yb == 1)).sum().item()
                fp += ((preds == 1) & (yb == 0)).sum().item()
                fn += ((preds == 0) & (yb == 1)).sum().item()

        acc  = correct / max(1, total)
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)

        print(f"{epoch:5d}  {avg_loss:8.4f}  {acc:7.4f}  {prec:7.4f}  {rec:7.4f}  {f1:7.4f}", end="")

        # Save best by F1 (balances precision/recall better than accuracy alone)
        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best_f1  = f1
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_accuracy": round(acc,  4),
                "val_f1":       round(f1,   4),
                "val_precision": round(prec, 4),
                "val_recall":   round(rec,  4),
            }, MODEL_OUT)
            print("  ← saved ✓", end="")

        print()

    print(f"\n{'=' * 50}")
    print(f"Done!  Best val   acc={best_acc:.4f}  f1={best_f1:.4f}")
    print(f"Model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
