"""
TruthLens AI - Experiment 1: Single-Modality vs Multimodal Comparison
Evaluates and compares:
1. Text-only Baseline (DistilBERT)
2. Image-only Baseline (ResNet-50)
3. Multimodal Model (Feature Fusion)
Outputs: Comprehensive comparison table, metrics JSON, and CSV.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.fusion.feature_fusion import FeatureFusionModel
from src.data.torch_dataset import create_multimodal_dataloader

OUTPUT_DIR = Path("experiments/exp1_single_vs_multimodal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, loader, model_type="multimodal"):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"]
            if model_type == "text":
                logits, _ = model(batch["input_ids"], batch["attention_mask"])
            elif model_type == "image":
                logits, _ = model(batch["image"])
            else:
                logits, _ = model(batch["image"], batch["input_ids"], batch["attention_mask"])

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.numpy())
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4)
    }


def run_experiment_1():
    print("=" * 60)
    print("Experiment 1: Single-Modality vs Multimodal Comparison")
    print("=" * 60)

    test_loader = create_multimodal_dataloader("data/splits/test.json", batch_size=16, is_training=False)

    CHECKPOINT_DIR = Path("checkpoints")

    # 1. Text Baseline
    print("Evaluating Text-Only Baseline...")
    text_model = TextBaselineModel()
    if (CHECKPOINT_DIR / "text_baseline.pt").exists():
        text_model.load_state_dict(torch.load(CHECKPOINT_DIR / "text_baseline.pt", map_location="cpu"))
        print("-> Loaded trained text_baseline.pt")
    text_metrics = evaluate_model(text_model, test_loader, model_type="text")

    # 2. Image Baseline
    print("Evaluating Image-Only Baseline...")
    image_model = ImageBaselineModel(pretrained=False)
    if (CHECKPOINT_DIR / "image_baseline.pt").exists():
        image_model.load_state_dict(torch.load(CHECKPOINT_DIR / "image_baseline.pt", map_location="cpu"))
        print("-> Loaded trained image_baseline.pt")
    image_metrics = evaluate_model(image_model, test_loader, model_type="image")

    # 3. Multimodal Model
    print("Evaluating Multimodal Model...")
    mm_model = FeatureFusionModel(text_backbone=text_model, image_backbone=image_model)
    if (CHECKPOINT_DIR / "feature_fusion.pt").exists():
        mm_model.load_state_dict(torch.load(CHECKPOINT_DIR / "feature_fusion.pt", map_location="cpu"))
        print("-> Loaded trained feature_fusion.pt")
    mm_metrics = evaluate_model(mm_model, test_loader, model_type="multimodal")

    results = [
        {"model": "Text-Only Baseline (DistilBERT)", **text_metrics},
        {"model": "Image-Only Baseline (ResNet-50)", **image_metrics},
        {"model": "Multimodal Model (Feature Fusion)", **mm_metrics}
    ]

    df = pd.DataFrame(results)
    print("\n--- Experiment 1 Results ---")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_DIR / "exp1_metrics.csv", index=False)
    with open(OUTPUT_DIR / "exp1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults successfully exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment_1()
