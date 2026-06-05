from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    losses = []
    all_true = []
    all_pred = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
    macro_f1 = _macro_f1(np.array(all_true), np.array(all_pred))
    return {"val_loss": float(np.mean(losses)) if losses else 0.0, "macro_f1": macro_f1}


def _write_history_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_training(
    model,
    train_ds,
    valid_ds,
    device: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    mixed_precision: bool,
    patience: int = 10,
    num_workers: int = 4,
    epoch_callback=None,
):
    """Train model. If valid_ds is None, trains for all epochs with no early stopping."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if valid_ds is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.startswith("cuda") else None
    model.to(device)

    best_metric = -1.0
    epochs_no_improve = 0
    best_path = out_path / "best.pt"
    history_csv = out_path / "history.csv"
    start = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        row = {"epoch": epoch, "train_loss": train_loss, "seconds": time.time() - start}
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metric": -1.0,
        }
        if valid_loader is not None:
            val_metrics = validate_one_epoch(model, valid_loader, device)
            row["val_loss"] = val_metrics["val_loss"]
            row["val_macro_f1"] = val_metrics["macro_f1"]
            ckpt["metric"] = val_metrics["macro_f1"]
            if val_metrics["macro_f1"] > best_metric:
                best_metric = val_metrics["macro_f1"]
                epochs_no_improve = 0
                torch.save(ckpt, best_path)
            else:
                epochs_no_improve += 1
            if epoch_callback is not None:
                epoch_callback(epoch, val_metrics["macro_f1"])
            if epochs_no_improve >= patience:
                break
        else:
            # No validation — save every epoch as best (last epoch wins)
            torch.save(ckpt, best_path)
        _write_history_row(history_csv, row)
        torch.save(ckpt, out_path / "last.pt")
    total_seconds = time.time() - start
    return {"best_macro_f1": best_metric, "history_csv": str(history_csv), "best_ckpt": str(best_path), "training_seconds": total_seconds}

