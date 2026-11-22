"""
Unit tests for Phase 5: Feature-Level Fusion and Decision-Level Fusion.
"""

import pytest
import torch

from src.fusion.feature_fusion import FeatureFusionModel
from src.fusion.decision_fusion import DecisionFusionEngine


def test_feature_fusion_forward_and_embeddings():
    """Verify FeatureFusionModel projects modalities and produces (B, 3) logits."""
    model = FeatureFusionModel(projection_dim=128)
    model.eval()

    # Synthetic batch of 2 samples
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_input_ids = torch.randint(100, 2000, (2, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((2, 16), dtype=torch.long)

    logits, embs = model(dummy_images, dummy_input_ids, dummy_attention_mask, return_embeddings=True)

    assert logits.shape == (2, 3)
    assert embs["proj_text"].shape == (2, 128)
    assert embs["proj_image"].shape == (2, 128)
    assert embs["fused_repr"].shape == (2, 128 * 6 + 1)

    probs = model.predict_proba(dummy_images, dummy_input_ids, dummy_attention_mask)
    assert probs.shape == (2, 3)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_decision_fusion_concordance():
    """Verify DecisionFusionEngine preserves class when both modalities strongly agree."""
    engine = DecisionFusionEngine()

    # Both text and image strongly predict FAKE (index 1)
    text_probs = torch.tensor([[0.05, 0.90, 0.05]])
    image_probs = torch.tensor([[0.08, 0.88, 0.04]])

    fused_probs, predicted_classes, diagnostics = engine.fuse_probabilities(text_probs, image_probs)
    
    assert predicted_classes[0].item() == 1  # FAKE
    assert fused_probs[0][1].item() > 0.80
    assert diagnostics["conflict_scores"][0] < 0.15


def test_decision_fusion_conflict_routes_to_uncertain():
    """Verify DecisionFusionEngine detects severe conflict and routes to UNCERTAIN (index 2)."""
    engine = DecisionFusionEngine(conflict_threshold=0.40)

    # Severe conflict: Text says REAL (0.95), Image says FAKE (0.95)
    text_probs = torch.tensor([[0.95, 0.03, 0.02]])
    image_probs = torch.tensor([[0.02, 0.95, 0.03]])

    fused_probs, predicted_classes, diagnostics = engine.fuse_probabilities(text_probs, image_probs)

    # Conflict score should be ~0.93
    assert diagnostics["conflict_scores"][0] > 0.80
    # The conflict penalty must elevate UNCERTAIN (index 2) as the top prediction
    assert predicted_classes[0].item() == 2  # UNCERTAIN
