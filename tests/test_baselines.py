"""
Unit tests for Phase 4: Text-Only and Image-Only Baselines.
"""

from pathlib import Path
import pytest
import torch

from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel


def test_text_baseline_forward_and_probabilities():
    """Verify TextBaselineModel outputs correct 3-class logits and probabilities."""
    model = TextBaselineModel()
    model.eval()

    # Batch of 2 synthetic text tokens (length 16)
    dummy_input_ids = torch.randint(100, 2000, (2, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((2, 16), dtype=torch.long)

    # Test forward with embeddings
    logits, features = model(dummy_input_ids, dummy_attention_mask, return_features=True)
    assert logits.shape == (2, 3), f"Expected logits (2, 3), got {logits.shape}"
    assert features.shape == (2, 768), f"Expected features (2, 768), got {features.shape}"

    # Test probability output
    probs = model.predict_proba(dummy_input_ids, dummy_attention_mask)
    assert probs.shape == (2, 3)
    # Assert probabilities sum to 1.0
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)


def test_image_baseline_forward_and_probabilities():
    """Verify ImageBaselineModel outputs correct 3-class logits and probabilities."""
    # Use uninitialized backbone for rapid unit testing without downloading ResNet weights
    model = ImageBaselineModel(pretrained=False)
    model.eval()

    # Batch of 2 dummy images (3, 224, 224)
    dummy_images = torch.randn(2, 3, 224, 224)

    logits, features = model(dummy_images, return_features=True)
    assert logits.shape == (2, 3), f"Expected logits (2, 3), got {logits.shape}"
    assert features.shape == (2, 2048), f"Expected features (2, 2048), got {features.shape}"

    probs = model.predict_proba(dummy_images)
    assert probs.shape == (2, 3)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
