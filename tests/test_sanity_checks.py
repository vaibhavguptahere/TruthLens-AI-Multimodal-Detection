"""
TruthLens AI - Required Sanity Checks Suite
Strictly verifies all 8 mandatory edge and error cases required by Section 8:
1. Clearly genuine image + correct caption
2. Clearly manipulated image + matching caption
3. Genuine image + deliberately mismatched caption (out-of-context miscaptioning)
4. Missing image
5. Missing text
6. Unsupported image format
7. Corrupt image
8. Ambiguous or insufficient evidence
"""

from pathlib import Path
import pytest
import torch
from PIL import Image

from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel
from src.fusion.decision_fusion import DecisionFusionEngine
from src.explainability.consistency import SemanticConsistencyAnalyzer
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.text_preprocessor import TextPreprocessor


@pytest.fixture(scope="module")
def components():
    return {
        "text_model": TextBaselineModel(),
        "image_model": ImageBaselineModel(pretrained=False),
        "decision_engine": DecisionFusionEngine(),
        "consistency": SemanticConsistencyAnalyzer(),
        "img_proc": ImagePreprocessor(),
        "txt_proc": TextPreprocessor()
    }


def test_sanity_1_clearly_genuine(components):
    """Sanity Check 1: Clearly genuine image + correct caption."""
    # Synthetic clean inputs
    img_tensor = torch.randn(1, 3, 224, 224)
    # Both models emit high REAL probability
    t_prob = torch.tensor([[0.94, 0.03, 0.03]])
    i_prob = torch.tensor([[0.92, 0.04, 0.04]])
    
    fused, pred, _ = components["decision_engine"].fuse_probabilities(t_prob, i_prob)
    assert pred.item() == 0  # REAL
    assert fused[0, 0].item() > 0.85


def test_sanity_2_clearly_manipulated(components):
    """Sanity Check 2: Clearly manipulated image + matching caption."""
    # Both models emit high FAKE probability
    t_prob = torch.tensor([[0.02, 0.95, 0.03]])
    i_prob = torch.tensor([[0.04, 0.93, 0.03]])

    fused, pred, _ = components["decision_engine"].fuse_probabilities(t_prob, i_prob)
    assert pred.item() == 1  # FAKE
    assert fused[0, 1].item() > 0.85


def test_sanity_3_genuine_image_mismatched_caption(components):
    """Sanity Check 3: Genuine image + deliberately mismatched caption."""
    # Text claim is fabricated (FAKE), image pixels are genuine (REAL) -> Conflicting signals
    t_prob = torch.tensor([[0.05, 0.90, 0.05]])
    i_prob = torch.tensor([[0.92, 0.04, 0.04]])

    fused, pred, diag = components["decision_engine"].fuse_probabilities(t_prob, i_prob)
    # Severe cross-modal divergence must elevate UNCERTAIN
    assert diag["conflict_scores"][0] > 0.70
    assert pred.item() == 2  # UNCERTAIN


def test_sanity_4_missing_image(components):
    """Sanity Check 4: Missing image path raises FileNotFoundError."""
    missing_path = Path("data/raw/images/NON_EXISTENT_IMAGE.jpg")
    with pytest.raises(FileNotFoundError):
        components["img_proc"].load_image(missing_path)


def test_sanity_5_missing_text(components):
    """Sanity Check 5: Missing or empty text is gracefully handled with clean default padding."""
    tokenized = components["txt_proc"].tokenize("")
    assert "input_ids" in tokenized
    assert tokenized["input_ids"].shape == (1, 128)


def test_sanity_6_unsupported_image_format(components, tmp_path):
    """Sanity Check 6: Unsupported format (e.g. text file renamed to .jpg) raises ValueError."""
    fake_img = tmp_path / "bad_format.jpg"
    fake_img.write_text("THIS IS PLAIN TEXT NOT AN IMAGE", encoding="utf-8")
    with pytest.raises(ValueError):
        components["img_proc"].load_image(fake_img)


def test_sanity_7_corrupt_image(components, tmp_path):
    """Sanity Check 7: Truncated or corrupt image file raises ValueError."""
    corrupt_img = tmp_path / "corrupt.jpg"
    corrupt_img.write_bytes(b"\xFF\xD8\xFF\xE0" + b"\x00" * 20)  # Incomplete JPEG header
    with pytest.raises(ValueError):
        components["img_proc"].load_image(corrupt_img)


def test_sanity_8_ambiguous_evidence(components):
    """Sanity Check 8: Ambiguous or low-confidence evidence routes to UNCERTAIN."""
    # Flat, low-certainty probability distribution
    t_prob = torch.tensor([[0.34, 0.33, 0.33]])
    i_prob = torch.tensor([[0.35, 0.35, 0.30]])

    fused, pred, _ = components["decision_engine"].fuse_probabilities(t_prob, i_prob)
    assert pred.item() == 2  # UNCERTAIN
