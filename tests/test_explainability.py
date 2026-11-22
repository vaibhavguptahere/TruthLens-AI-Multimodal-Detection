"""
Unit tests for Phase 6: Explainability & Uncertainty Verification.
"""

import numpy as np
import pytest
import torch

from src.models.image_baseline import ImageBaselineModel
from src.models.text_baseline import TextBaselineModel
from src.explainability.gradcam import GradCAMExplainer
from src.explainability.text_saliency import TextSaliencyExplainer
from src.explainability.consistency import SemanticConsistencyAnalyzer
from src.explainability.report_generator import VerificationReportGenerator


def test_gradcam_heatmap_generation():
    """Verify Grad-CAM produces a valid 2D normalized (224, 224) heatmap."""
    model = ImageBaselineModel(pretrained=False)
    explainer = GradCAMExplainer(model=model)

    dummy_img = torch.randn(1, 3, 224, 224)
    heatmap = explainer.generate_heatmap(dummy_img, target_class=1)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    explainer.cleanup()


def test_text_saliency_attribution():
    """Verify text explainer attributes importance scores to individual words."""
    model = TextBaselineModel()
    explainer = TextSaliencyExplainer(model=model)

    text = "Breaking fake news reporting fabricated event."
    attributions = explainer.explain(text, target_class=1)

    assert len(attributions) > 0
    for item in attributions:
        assert "token" in item
        assert "importance" in item
        assert 0.0 <= item["importance"] <= 1.0


def test_semantic_consistency_analyzer():
    """Verify cross-modal cosine similarity and categorization."""
    analyzer = SemanticConsistencyAnalyzer(aligned_threshold=0.6, misaligned_threshold=0.2)

    # Identical vectors -> High cosine similarity (ALIGNED)
    v1 = torch.tensor([[1.0, 0.0, 0.0]])
    v2 = torch.tensor([[1.0, 0.0, 0.0]])
    res_aligned = analyzer.compute_consistency(v1, v2)
    assert res_aligned["consistency_status"] == "ALIGNED"
    assert res_aligned["cosine_similarity"] >= 0.99

    # Orthogonal vectors -> Low cosine similarity (MISALIGNED)
    v3 = torch.tensor([[0.0, 1.0, 0.0]])
    res_misaligned = analyzer.compute_consistency(v1, v3)
    assert res_misaligned["consistency_status"] == "MISALIGNED"
    assert res_misaligned["cosine_similarity"] <= 0.05


def test_verification_report_generation():
    """Verify that VerificationReportGenerator builds structured JSON with all audit trails."""
    report = VerificationReportGenerator.generate_report(
        prediction="FAKE",
        confidence=0.88,
        text_prediction="FAKE",
        text_confidence=0.82,
        image_prediction="FAKE",
        image_confidence=0.91,
        consistency_data={"consistency_status": "ALIGNED", "cosine_similarity": 0.75},
        top_text_tokens=[{"token": "fabricated", "importance": 0.95}],
        sample_id="TL-TEST-001"
    )

    assert report["final_verdict"] == "FAKE"
    assert report["confidence_score"] == 0.88
    assert report["calibrated_status"] == "HIGH_CONFIDENCE"
    assert "Visual analysis detected structural pixel manipulation" in report["evidence_explanation"]
    assert "modality_breakdown" in report
