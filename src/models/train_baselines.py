"""
TruthLens AI - Anti-Overfitting Baseline Training Engine
Features:
- AdamW optimizer with L2 weight decay (1e-2)
- Cosine Annealing Learning Rate Scheduler with Warmup
- Validation Loss Early Stopping (patience=3)
- Checkpoint saving strictly based on best validation Macro-F1 score
- Training & validation loss tracking exported to JSON
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.data.torch_dataset import create_multimodal_dataloader

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class EarlyStopping:
    """Early stops training when validation loss stops improving to prevent overfitting."""
    def __init__(self, patience: int = 3, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            return True  # Model improved
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # No improvement
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            return True  # Model improved


def train_epoch_text(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_text(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return {
        "val_loss": total_loss / len(loader),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1)
    }


def train_epoch_image(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_image(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return {
        "val_loss": total_loss / len(loader),
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1)
    }


def train_single_modality_baselines(max_epochs: int = 10, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"TruthLens AI: Training Baselines with Anti-Overfitting Protocol")
    print(f"Device: {device} | Optimizer: AdamW (weight_decay=0.01)")
    print(f"========================================================")

    train_loader = create_multimodal_dataloader("data/splits/train.json", batch_size=batch_size, is_training=True)
    val_loader = create_multimodal_dataloader("data/splits/val.json", batch_size=batch_size, is_training=False)
    test_loader = create_multimodal_dataloader("data/splits/test.json", batch_size=batch_size, is_training=False)

    # Label smoothing in loss function to prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 1. TEXT BASELINE TRAINING
    print("\n--- [1/2] Fine-Tuning Text-Only Baseline (DistilBERT) ---")
    text_model = TextBaselineModel(dropout_prob=0.3).to(device)
    
    # Differential learning rate: freeze bottom transformer layers (0 to 3),
    # unfreeze top 2 transformer layers (layer 4 and 5) for domain adaptation
    for param in text_model.transformer.parameters():
        param.requires_grad = False

    # Unfreeze top layers in DistilBERT transformer
    for name, param in text_model.transformer.named_parameters():
        if any(f"layer.{i}" in name for i in [4, 5]):
            param.requires_grad = True

    # Parameter groups with different learning rates
    backbone_params = [p for p in text_model.transformer.parameters() if p.requires_grad]
    head_params = list(text_model.classifier.parameters())

    optimizer_txt = AdamW([
        {"params": backbone_params, "lr": 1.5e-5},      # Gentle LR for transformer layers
        {"params": head_params, "lr": 3e-4, "weight_decay": 1e-2}  # Standard LR for classification head
    ])
    scheduler_txt = CosineAnnealingLR(optimizer_txt, T_max=max_epochs)
    early_stopping_txt = EarlyStopping(patience=3)

    best_txt_f1 = 0.0
    txt_history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch_text(text_model, train_loader, optimizer_txt, criterion, device)
        val_res = evaluate_text(text_model, val_loader, criterion, device)
        scheduler_txt.step()

        improved = early_stopping_txt(val_res["val_loss"])
        if improved and val_res["f1"] > best_txt_f1:
            best_txt_f1 = val_res["f1"]
            torch.save(text_model.state_dict(), CHECKPOINT_DIR / "text_baseline.pt")
            saved_tag = "[BEST - Checkpoint Saved]"
        else:
            saved_tag = ""

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Acc: {val_res['accuracy']:.4f} | Val F1: {val_res['f1']:.4f} {saved_tag}")
        txt_history.append({"epoch": epoch, "train_loss": train_loss, **val_res})

        if early_stopping_txt.early_stop:
            print(f"-> Early stopping triggered at epoch {epoch} to prevent overfitting.")
            break

    # Load best checkpoint for test evaluation
    text_model.load_state_dict(torch.load(CHECKPOINT_DIR / "text_baseline.pt", map_location=device))
    test_txt_res = evaluate_text(text_model, test_loader, criterion, device)
    print(f"\n-> Text Baseline Final Test Results (Best Model): {test_txt_res}")

    # 2. IMAGE BASELINE TRAINING
    print("\n--- [2/2] Fine-Tuning Image-Only Baseline (ResNet-50) ---")
    image_model = ImageBaselineModel(pretrained=True, dropout_prob=0.3).to(device)

    # Freeze earlier conv layers (conv1, bn1, layer1, layer2, layer3)
    for param in image_model.conv_stem.parameters():
        param.requires_grad = False

    # Unfreeze layer4 (the highest-level feature bottleneck at conv_stem[7]) for fine-tuning
    for param in image_model.conv_stem[7].parameters():
        param.requires_grad = True

    layer4_params = list(image_model.conv_stem[7].parameters())
    img_head_params = list(image_model.classifier.parameters())

    optimizer_img = AdamW([
        {"params": layer4_params, "lr": 1e-5},           # Gentle LR for visual features
        {"params": img_head_params, "lr": 3e-4, "weight_decay": 1e-2}  # Standard LR for head
    ])
    scheduler_img = CosineAnnealingLR(optimizer_img, T_max=max_epochs)
    early_stopping_img = EarlyStopping(patience=3)

    best_img_f1 = 0.0
    img_history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch_image(image_model, train_loader, optimizer_img, criterion, device)
        val_res = evaluate_image(image_model, val_loader, criterion, device)
        scheduler_img.step()

        improved = early_stopping_img(val_res["val_loss"])
        if improved and val_res["f1"] > best_img_f1:
            best_img_f1 = val_res["f1"]
            torch.save(image_model.state_dict(), CHECKPOINT_DIR / "image_baseline.pt")
            saved_tag = "[BEST - Checkpoint Saved]"
        else:
            saved_tag = ""

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Acc: {val_res['accuracy']:.4f} | Val F1: {val_res['f1']:.4f} {saved_tag}")
        img_history.append({"epoch": epoch, "train_loss": train_loss, **val_res})

        if early_stopping_img.early_stop:
            print(f"-> Early stopping triggered at epoch {epoch} to prevent overfitting.")
            break

    image_model.load_state_dict(torch.load(CHECKPOINT_DIR / "image_baseline.pt", map_location=device))
    test_img_res = evaluate_image(image_model, test_loader, criterion, device)
    print(f"\n-> Image Baseline Final Test Results (Best Model): {test_img_res}")

    # Export training histories
    with open(CHECKPOINT_DIR / "baseline_training_history.json", "w", encoding="utf-8") as f:
        json.dump({"text_baseline": txt_history, "image_baseline": img_history}, f, indent=2)


if __name__ == "__main__":
    train_single_modality_baselines(max_epochs=8)
