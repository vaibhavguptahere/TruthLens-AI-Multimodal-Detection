"""
TruthLens AI - Experiment 5: Difficult-Case Analysis
Isolates and performs in-depth qualitative and quantitative analysis of at least 15
challenging corner cases (semantic mismatches, misleading captions, ambiguous artifacts, etc.).
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import pandas as pd
import torch

from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.fusion.decision_fusion import DecisionFusionEngine
from src.explainability.consistency import SemanticConsistencyAnalyzer
from src.explainability.report_generator import VerificationReportGenerator
from src.data.torch_dataset import MultimodalDataset, REV_LABEL_MAP

OUTPUT_DIR = Path("experiments/exp5_difficult_cases")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment_5():
    print("=" * 60)
    print("Experiment 5: 15 Difficult / Corner Cases Analysis")
    print("=" * 60)

    dataset = MultimodalDataset("data/splits/test.json", is_training=False)
    CHECKPOINT_DIR = Path("checkpoints")
    text_model = TextBaselineModel()
    if (CHECKPOINT_DIR / "text_baseline.pt").exists():
        text_model.load_state_dict(torch.load(CHECKPOINT_DIR / "text_baseline.pt", map_location="cpu"))
    
    image_model = ImageBaselineModel(pretrained=False)
    if (CHECKPOINT_DIR / "image_baseline.pt").exists():
        image_model.load_state_dict(torch.load(CHECKPOINT_DIR / "image_baseline.pt", map_location="cpu"))
    decision_engine = DecisionFusionEngine()
    consistency_analyzer = SemanticConsistencyAnalyzer()

    # We use FeatureFusionModel to get aligned projected embeddings
    from src.fusion.feature_fusion import FeatureFusionModel
    fusion_model = FeatureFusionModel(text_backbone=text_model, image_backbone=image_model)
    text_model.eval()
    image_model.eval()
    fusion_model.eval()
    
    for i, item in enumerate(dataset):
        if len(difficult_cases) >= 15:
            break

        m_type = item["manipulation_type"]
        is_difficult = (
            m_type in ["out_of_context", "ambiguous", "photoshop_splice"]
            or item["label"].item() == 2  # UNCERTAIN
        )

        if is_difficult:
            # Inference
            with torch.no_grad():
                img = item["image"].unsqueeze(0)
                ids = item["input_ids"].unsqueeze(0)
                mask = item["attention_mask"].unsqueeze(0)

                t_prob = text_model.predict_proba(ids, mask)
                i_prob = image_model.predict_proba(img)
                fused_prob, pred_idx, diag = decision_engine.fuse_probabilities(t_prob, i_prob)

                # Consistency using aligned projections
                p_txt, p_img, _ = fusion_model.extract_multimodal_embeddings(img, ids, mask)
                cons_data = consistency_analyzer.compute_consistency(p_txt, p_img)

            pred_label = REV_LABEL_MAP[pred_idx.item()]
            gt_label = REV_LABEL_MAP[item["label"].item()]
            conf = float(fused_prob[0, pred_idx.item()].item())

            report = VerificationReportGenerator.generate_report(
                prediction=pred_label,
                confidence=conf,
                text_prediction=REV_LABEL_MAP[torch.argmax(t_prob).item()],
                text_confidence=float(torch.max(t_prob).item()),
                image_prediction=REV_LABEL_MAP[torch.argmax(i_prob).item()],
                image_confidence=float(torch.max(i_prob).item()),
                consistency_data=cons_data,
                top_text_tokens=[],
                sample_id=item["sample_id"]
            )

            case_record = {
                "case_num": len(difficult_cases) + 1,
                "sample_id": item["sample_id"],
                "manipulation_type": m_type,
                "ground_truth": gt_label,
                "model_prediction": pred_label,
                "confidence": round(conf, 4),
                "conflict_score": round(diag["conflict_scores"][0], 4),
                "consistency_status": cons_data["consistency_status"],
                "explanation": report["evidence_explanation"]
            }
            difficult_cases.append(case_record)

    df = pd.DataFrame(difficult_cases)
    print(f"\n--- Analyzed {len(difficult_cases)} Difficult Cases ---")
    print(df[["case_num", "sample_id", "manipulation_type", "ground_truth", "model_prediction", "confidence"]].to_string(index=False))

    df.to_csv(OUTPUT_DIR / "difficult_cases_analysis.csv", index=False)
    with open(OUTPUT_DIR / "difficult_cases_analysis.json", "w", encoding="utf-8") as f:
        json.dump(difficult_cases, f, indent=2)

    print(f"\nResults successfully exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_experiment_5()
