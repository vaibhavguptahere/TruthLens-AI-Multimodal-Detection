"""
TruthLens AI - Unified Multimodal Inference Engine
Wraps text preprocessor, image preprocessor, feature fusion, decision fusion,
Grad-CAM, text saliency, and report generator into a single callable pipeline.
"""

import base64
import io
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
import numpy as np
import torch

from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.fusion.feature_fusion import FeatureFusionModel
from src.fusion.decision_fusion import DecisionFusionEngine
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.text_saliency import TextSaliencyExplainer
from src.explainability.consistency import SemanticConsistencyAnalyzer
from src.explainability.report_generator import VerificationReportGenerator
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.data.torch_dataset import REV_LABEL_MAP


class TruthLensPipeline:
    """
    End-to-end multimodal verification pipeline for TruthLens AI.
    """
    def __init__(self):
        self.img_proc = ImagePreprocessor()
        self.txt_proc = TextPreprocessor()

        CHECKPOINT_DIR = Path("checkpoints")
        self.text_model = TextBaselineModel()
        if (CHECKPOINT_DIR / "text_baseline.pt").exists():
            self.text_model.load_state_dict(torch.load(CHECKPOINT_DIR / "text_baseline.pt", map_location="cpu"))

        self.image_model = ImageBaselineModel(pretrained=False)
        if (CHECKPOINT_DIR / "image_baseline.pt").exists():
            self.image_model.load_state_dict(torch.load(CHECKPOINT_DIR / "image_baseline.pt", map_location="cpu"))

        self.feature_fusion = FeatureFusionModel(
            text_backbone=self.text_model,
            image_backbone=self.image_model
        )
        if (CHECKPOINT_DIR / "feature_fusion.pt").exists():
            self.feature_fusion.load_state_dict(torch.load(CHECKPOINT_DIR / "feature_fusion.pt", map_location="cpu"))
        self.decision_engine = DecisionFusionEngine()
        self.consistency_analyzer = SemanticConsistencyAnalyzer()

        # Explainers
        self.gradcam = GradCAMExplainer(self.image_model)
        self.saliency = TextSaliencyExplainer(self.text_model)

    def verify(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        caption_text: str
    ) -> Dict[str, Any]:
        """
        Executes end-to-end verification and returns prediction, confidence,
        Grad-CAM heatmap base64, token attributions, and structured report.
        """
        # 1. Process inputs
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            pil_img = self.img_proc.load_image(image_input)

        img_tensor = self.img_proc.preprocess(pil_img).unsqueeze(0)
        tokenized = self.txt_proc.tokenize(caption_text)
        ids = tokenized["input_ids"]
        mask = tokenized["attention_mask"]

        # 2. Single-modality probabilities
        t_prob = self.text_model.predict_proba(ids, mask)
        i_prob = self.image_model.predict_proba(img_tensor)

        # 3. Decision-Level Fusion & Conflict Routing
        fused_prob, pred_idx, diag = self.decision_engine.fuse_probabilities(t_prob, i_prob)
        prediction_str = REV_LABEL_MAP[pred_idx.item()]
        confidence_val = float(fused_prob[0, pred_idx.item()].item())

        text_pred = REV_LABEL_MAP[torch.argmax(t_prob).item()]
        text_conf = float(torch.max(t_prob).item())
        img_pred = REV_LABEL_MAP[torch.argmax(i_prob).item()]
        img_conf = float(torch.max(i_prob).item())

        # 4. Cross-Modal Semantic Consistency
        with torch.no_grad():
            p_txt, p_img, _ = self.feature_fusion.extract_multimodal_embeddings(img_tensor, ids, mask)
            consistency_data = self.consistency_analyzer.compute_consistency(p_txt, p_img)

        # 5. Explainability (Grad-CAM Heatmap & Text Saliency)
        heatmap_2d = self.gradcam.generate_heatmap(img_tensor, target_class=pred_idx.item())
        
        # Colorize and encode Grad-CAM as base64 PNG
        heatmap_uint8 = np.uint8(255 * heatmap_2d)
        heatmap_pil = Image.fromarray(heatmap_uint8).resize(pil_img.size)
        buffer = io.BytesIO()
        heatmap_pil.save(buffer, format="PNG")
        heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Text Saliency
        token_attributions = self.saliency.explain(caption_text, target_class=pred_idx.item())

        # 6. Generate Verification Report
        report = VerificationReportGenerator.generate_report(
            prediction=prediction_str,
            confidence=confidence_val,
            text_prediction=text_pred,
            text_confidence=text_conf,
            image_prediction=img_pred,
            image_confidence=img_conf,
            consistency_data=consistency_data,
            top_text_tokens=token_attributions
        )

        return {
            "prediction": prediction_str,
            "confidence": round(confidence_val, 4),
            "conflict_score": round(diag["conflict_scores"][0], 4),
            "text_prediction": text_pred,
            "text_confidence": round(text_conf, 4),
            "image_prediction": img_pred,
            "image_confidence": round(img_conf, 4),
            "consistency": consistency_data,
            "token_attributions": token_attributions[:10],
            "gradcam_heatmap_base64": heatmap_b64,
            "report": report
        }
