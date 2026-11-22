"""
TruthLens AI - Experiment 2: Fusion Comparison
Directly compares Feature-Level Fusion against Decision-Level Fusion on the test set.
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
from src.fusion.decision_fusion import DecisionFusionEngine
from src.data.torch_dataset import create_multimodal_dataloader

OUTPUT_DIR = Path("experiments/exp2_fusion_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment_2():
    print("=" * 60)
    print("Experiment 2: Feature-Level vs Decision-Level Fusion")
    print("=" * 60)

    test_loader = create_multimodal_dataloader("data/splits/test.json", batch_size=16, is_training=False)

    CHECKPOINT_DIR = Path("checkpoints")
    text_model = TextBaselineModel()
    if (CHECKPOINT_DIR / "text_baseline.pt").exists():
        text_model.load_state_dict(torch.load(CHECKPOINT_DIR / "text_baseline.pt", map_location="cpu"))

    image_model = ImageBaselineModel(pretrained=False)
    if (CHECKPOINT_DIR / "image_baseline.pt").exists():
        image_model.load_state_dict(torch.load(CHECKPOINT_DIR / "image_baseline.pt", map_location="cpu"))

    feature_model = FeatureFusionModel(text_backbone=text_model, image_backbone=image_model)
    if (CHECKPOINT_DIR / "feature_fusion.pt").exists():
        feature_model.load_state_dict(torch.load(CHECKPOINT_DIR / "feature_fusion.pt", map_location="cpu"))

    text_model.eval()
    image_model.eval()
    feature_model.eval()

    decision_engine = DecisionFusionEngine()

    all_targets = []
    feat_preds = []
    dec_preds = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch["label"]
            all_targets.extend(labels.numpy())

            # Feature Fusion prediction
            f_logits, _ = feature_model(batch["image"], batch["input_ids"], batch["attention_mask"])
            feat_preds.extend(torch.argmax(f_logits, dim=-1).numpy())

            # Decision Fusion prediction
            t_probs = text_model.predict_proba(batch["input_ids"], batch["attention_mask"])
            i_probs = image_model.predict_proba(batch["image"])
            _, d_pred, _ = decision_engine.fuse_probabilities(t_probs, i_probs)
            dec_preds.extend(d_pred.numpy())

    # Metrics
    f_acc = accuracy_score(all_targets, feat_preds)
    f_p, f_r, f_f1, _ = precision_recall_fscore_support(all_targets, feat_preds, average="macro", zero_division=0)

    d_acc = accuracy_score(all_targets, dec_preds)
    d_p, d_r, d_f1, _ = precision_recall_fscore_support(all_targets, dec_preds, average="macro", zero_division=0)

    results = [
        {
            "fusion_strategy": "Feature-Level Fusion",
            "accuracy": round(float(f_acc), 4),
            "precision": round(float(f_p), 4),
            "recall": round(float(f_r), 4),
            "f1": round(float(f_f1), 4)
        },
        {
            "fusion_strategy": "Decision-Level Fusion (Conflict-Aware)",
            "accuracy": round(float(d_acc), 4),
            "precision": round(float(d_p), 4),
            "recall": round(float(d_r), 4),
            "f1": round(float(d_f1), 4)
        }
    ]

    df = pd.DataFrame(results)
    print("\n--- Experiment 2 Results ---")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_DIR / "exp2_metrics.csv", index=False)
    with open(OUTPUT_DIR / "exp2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults successfully exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment_2()
