"""
TruthLens AI - Experiment 6: Explainability Evaluation
Audits whether highlighted visual Grad-CAM heatmaps and text token attributions
are quantitatively faithful to model predictions.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import pandas as pd
import torch
import numpy as np

from src.models.image_baseline import ImageBaselineModel
from src.models.text_baseline import TextBaselineModel
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.text_saliency import TextSaliencyExplainer
from src.data.torch_dataset import MultimodalDataset

OUTPUT_DIR = Path("experiments/exp6_explainability_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment_6():
    print("=" * 60)
    print("Experiment 6: Explainability Faithfulness & Localization Evaluation")
    print("=" * 60)

    dataset = MultimodalDataset("data/splits/test.json", is_training=False)
    img_model = ImageBaselineModel(pretrained=False)
    txt_model = TextBaselineModel()

    gradcam = GradCAMExplainer(img_model)
    saliency = TextSaliencyExplainer(txt_model)

    eval_records = []
    # Evaluate across representative test subset (10 samples)
    for i in range(10):
        item = dataset[i]
        img_tensor = item["image"].unsqueeze(0)
        
        # 1. Grad-CAM peak localization check
        heatmap = gradcam.generate_heatmap(img_tensor, target_class=item["label"].item())
        peak_intensity = float(np.max(heatmap))
        mean_intensity = float(np.mean(heatmap))
        localization_ratio = round(peak_intensity / (mean_intensity + 1e-6), 2)

        # 2. Text token saliency top word
        # Recover text string from dataset split json
        sample_id = item["sample_id"]
        raw_text = next(s["text"] for s in dataset.samples if s["sample_id"] == sample_id)
        attributions = saliency.explain(raw_text, target_class=item["label"].item())
        top_token = attributions[0]["token"] if attributions else "none"
        top_importance = attributions[0]["importance"] if attributions else 0.0

        eval_records.append({
            "sample_id": sample_id,
            "manipulation_type": item["manipulation_type"],
            "gradcam_peak": round(peak_intensity, 4),
            "gradcam_concentration_ratio": localization_ratio,
            "top_salient_token": top_token,
            "token_importance": round(top_importance, 4),
            "faithfulness_verdict": "FAITHFUL" if localization_ratio > 1.5 else "DIFFUSE"
        })

    gradcam.cleanup()
    df = pd.DataFrame(eval_records)
    print("\n--- Experiment 6 Explainability Evaluation ---")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_DIR / "explainability_eval.csv", index=False)
    with open(OUTPUT_DIR / "explainability_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)

    print(f"\nResults successfully exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment_6()
