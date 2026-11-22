"""
TruthLens AI - Anti-Overfitting Multimodal Fusion Training Engine
Features:
- Freezes base unimodal representations to avoid representation distortion
- Trains bottleneck projections and joint classification head
- High dropout (p=0.4), LayerNorm, and AdamW weight decay (1e-2)
- Early stopping based on validation loss
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.fusion.feature_fusion import FeatureFusionModel
from src.data.torch_dataset import create_multimodal_dataloader

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class EarlyStopping:
    def __init__(self, patience: int = 3, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.counter = 0
            return True


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(images, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(images, input_ids, attention_mask)
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


def train_multimodal_fusion(max_epochs: int = 10, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"TruthLens AI: Training Feature-Level Fusion (Anti-Overfitting)")
    print(f"Device: {device} | Optimizer: AdamW (weight_decay=0.01)")
    print(f"========================================================")

    train_loader = create_multimodal_dataloader("data/splits/train.json", batch_size=batch_size, is_training=True)
    val_loader = create_multimodal_dataloader("data/splits/val.json", batch_size=batch_size, is_training=False)
    test_loader = create_multimodal_dataloader("data/splits/test.json", batch_size=batch_size, is_training=False)

    # Load frozen pretrained encoders
    text_model = TextBaselineModel(dropout_prob=0.4).to(device)
    if (CHECKPOINT_DIR / "text_baseline.pt").exists():
        text_model.load_state_dict(torch.load(CHECKPOINT_DIR / "text_baseline.pt", map_location=device))
        print("Loaded trained text baseline checkpoint.")

    image_model = ImageBaselineModel(pretrained=False, dropout_prob=0.4).to(device)
    if (CHECKPOINT_DIR / "image_baseline.pt").exists():
        image_model.load_state_dict(torch.load(CHECKPOINT_DIR / "image_baseline.pt", map_location=device))
        print("Loaded trained image baseline checkpoint.")

    fusion_model = FeatureFusionModel(
        text_backbone=text_model,
        image_backbone=image_model,
        dropout_prob=0.2
    ).to(device)

    # Freeze lower layers, unfreeze top representations for fine-tuning
    for p in fusion_model.text_model.parameters():
        p.requires_grad = False
    for p in fusion_model.image_model.parameters():
        p.requires_grad = False

    # Unfreeze top transformer layer (layer 5) and top conv block (layer4)
    for name, p in fusion_model.text_model.transformer.named_parameters():
        if "layer.5" in name:
            p.requires_grad = True

    for p in fusion_model.image_model.conv_stem[7].parameters():
        p.requires_grad = True

    encoder_params = [p for p in fusion_model.text_model.parameters() if p.requires_grad] + \
                     [p for p in fusion_model.image_model.parameters() if p.requires_grad]

    fusion_head_params = list(fusion_model.text_proj.parameters()) + \
                         list(fusion_model.image_proj.parameters()) + \
                         list(fusion_model.classifier.parameters())

    optimizer = AdamW([
        {"params": encoder_params, "lr": 2e-5},
        {"params": fusion_head_params, "lr": 5e-4, "weight_decay": 5e-3}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    
    # Class weights to address UNCERTAIN (56) vs REAL (159) vs FAKE (175) imbalance:
    # 0: REAL, 1: FAKE, 2: UNCERTAIN
    class_weights = torch.tensor([1.1, 1.0, 2.5], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    early_stopping = EarlyStopping(patience=4)

    best_f1 = 0.0
    fusion_history = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(fusion_model, train_loader, optimizer, criterion, device)
        val_res = evaluate(fusion_model, val_loader, criterion, device)
        scheduler.step()

        improved = early_stopping(val_res["val_loss"])
        if improved and val_res["f1"] > best_f1:
            best_f1 = val_res["f1"]
            torch.save(fusion_model.state_dict(), CHECKPOINT_DIR / "feature_fusion.pt")
            saved_tag = "[BEST - Checkpoint Saved]"
        else:
            saved_tag = ""

        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_res['val_loss']:.4f} | Val Acc: {val_res['accuracy']:.4f} | Val F1: {val_res['f1']:.4f} {saved_tag}")
        fusion_history.append({"epoch": epoch, "train_loss": train_loss, **val_res})

        if early_stopping.early_stop:
            print(f"-> Early stopping triggered at epoch {epoch} to prevent overfitting.")
            break

    # Load best model for test evaluation
    fusion_model.load_state_dict(torch.load(CHECKPOINT_DIR / "feature_fusion.pt", map_location=device))
    test_res = evaluate(fusion_model, test_loader, criterion, device)
    print(f"\n-> Multimodal Fusion Final Test Results (Best Model): {test_res}")

    with open(CHECKPOINT_DIR / "fusion_training_history.json", "w", encoding="utf-8") as f:
        json.dump(fusion_history, f, indent=2)


if __name__ == "__main__":
    train_multimodal_fusion(max_epochs=8)
