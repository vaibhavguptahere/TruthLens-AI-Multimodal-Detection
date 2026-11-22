"""
TruthLens AI - Experiment 4: Modality Ablation Study
Ablates inputs to systematically quantify the contribution of each modality:
1. Full Multimodal (Image + Text)
2. Image Only (Text zeroed out)
3. Text Only (Image zeroed out)
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

from src.fusion.feature_fusion import FeatureFusionModel
from src.data.torch_dataset import create_multimodal_dataloader

OUTPUT_DIR = Path("experiments/exp4_modality_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment_4():
    print("=" * 60)
    print("Experiment 4: Modality Ablation Study")
    print("=" * 60)

    CHECKPOINT_DIR = Path("checkpoints")
    test_loader = create_multimodal_dataloader("data/splits/test.json", batch_size=16, is_training=False)
    model = FeatureFusionModel()
    if (CHECKPOINT_DIR / "feature_fusion.pt").exists():
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "feature_fusion.pt", map_location="cpu"))
    model.eval()

    conditions = ["Full Multimodal (Image + Text)", "Image Only (Ablated Text)", "Text Only (Ablated Image)"]
    all_results = []

    for condition in conditions:
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                labels = batch["label"]
                images = batch["image"]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                if "Image Only" in condition:
                    # Ablate text: zero out attention mask
                    attention_mask = torch.zeros_like(attention_mask)
                elif "Text Only" in condition:
                    # Ablate image: zero out image tensor
                    images = torch.zeros_like(images)

                logits, _ = model(images, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.numpy())
                all_targets.extend(labels.numpy())

        acc = accuracy_score(all_targets, all_preds)
        p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
        all_results.append({
            "ablation_condition": condition,
            "accuracy": round(float(acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4)
        })

    df = pd.DataFrame(all_results)
    print("\n--- Experiment 4 Results ---")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_DIR / "exp4_ablation_metrics.csv", index=False)
    with open(OUTPUT_DIR / "exp4_ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults successfully exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment_4()
