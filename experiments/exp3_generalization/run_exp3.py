"""
TruthLens AI - Experiment 3: Generalization Analysis
Evaluates multimodal detection performance broken down by individual manipulation type:
- Diffusion Generated
- Face Swap
- Photoshop Splicing
- Out-of-Context (Semantic Miscaptioning)
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from src.fusion.feature_fusion import FeatureFusionModel
from src.data.torch_dataset import MultimodalDataset

OUTPUT_DIR = Path("experiments/exp3_generalization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment_3():
    print("=" * 60)
    print("Experiment 3: Generalization Across Manipulation Types")
    print("=" * 60)

    dataset = MultimodalDataset("data/splits/test.json", is_training=False)
    CHECKPOINT_DIR = Path("checkpoints")
    model = FeatureFusionModel()
    if (CHECKPOINT_DIR / "feature_fusion.pt").exists():
        model.load_state_dict(torch.load(CHECKPOINT_DIR / "feature_fusion.pt", map_location="cpu"))
    model.eval()

    by_type = {}
    for item in dataset:
        m_type = item["manipulation_type"]
        if m_type not in by_type:
            by_type[m_type] = {"targets": [], "preds": []}

        with torch.no_grad():
            img = item["image"].unsqueeze(0)
            ids = item["input_ids"].unsqueeze(0)
            mask = item["attention_mask"].unsqueeze(0)
            logits, _ = model(img, ids, mask)
            pred = torch.argmax(logits, dim=-1).item()

        by_type[m_type]["targets"].append(item["label"].item())
        by_type[m_type]["preds"].append(pred)

    results = []
    for m_type, data in by_type.items():
        acc = accuracy_score(data["targets"], data["preds"])
        results.append({
            "manipulation_type": m_type,
            "sample_count": len(data["targets"]),
            "accuracy": round(float(acc), 4)
        })

    df = pd.DataFrame(results)
    print("\n--- Experiment 3 Results ---")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_DIR / "exp3_generalization_metrics.csv", index=False)
    with open(OUTPUT_DIR / "exp3_generalization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults successfully exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment_3()
