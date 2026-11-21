"""
Tests for Phase 2: Dataset Integrity, Schema Validation, and Metadata Consistency.
"""

import json
from pathlib import Path
import pytest
from PIL import Image

from src.data.schema import (
    MultimodalSampleMetadata,
    PairLabel,
    ImageAuthenticity,
    TextClaimStatus,
    ImageTextConsistency,
    ManipulationType
)
from src.data.labeling_protocol import determine_pair_label

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
RAW_IMAGES_DIR = Path("data/raw/images")


def test_dataset_files_exist():
    """Verify that all required split and master dataset files are generated."""
    assert (PROCESSED_DIR / "dataset_master.json").exists()
    assert (PROCESSED_DIR / "dataset_master.csv").exists()
    assert (SPLITS_DIR / "train.json").exists()
    assert (SPLITS_DIR / "val.json").exists()
    assert (SPLITS_DIR / "test.json").exists()


def test_schema_conformance_and_images_valid():
    """Verify each sample conforms to Pydantic schema and references a valid readable image."""
    with open(PROCESSED_DIR / "dataset_master.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 450, f"Expected 450 samples, found {len(records)}"

    for record in records:
        # Schema validation
        sample = MultimodalSampleMetadata(**record)
        
        # Image existence and readability check
        img_full_path = Path(sample.image_rel_path)
        assert img_full_path.exists(), f"Image does not exist: {img_full_path}"
        with Image.open(img_full_path) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"


def test_labeling_protocol_logic():
    """Verify deterministic mapping in labeling_protocol."""
    # Authentic clean sample -> REAL
    assert determine_pair_label(
        ImageAuthenticity.AUTHENTIC,
        TextClaimStatus.VERIFIABLE_TRUE,
        ImageTextConsistency.ALIGNED
    ) == PairLabel.REAL

    # Manipulated image -> FAKE
    assert determine_pair_label(
        ImageAuthenticity.MANIPULATED,
        TextClaimStatus.FABRICATED,
        ImageTextConsistency.ALIGNED
    ) == PairLabel.FAKE

    # Authentic image + Fabricated caption + Misaligned -> FAKE (out-of-context miscaptioning)
    assert determine_pair_label(
        ImageAuthenticity.AUTHENTIC,
        TextClaimStatus.FABRICATED,
        ImageTextConsistency.MISALIGNED
    ) == PairLabel.FAKE

    # Ambiguous / Unknown evidence -> UNCERTAIN
    assert determine_pair_label(
        ImageAuthenticity.UNKNOWN,
        TextClaimStatus.UNKNOWN,
        ImageTextConsistency.AMBIGUOUS
    ) == PairLabel.UNCERTAIN


def test_stratified_split_balance():
    """Verify that train/val/test splits maintain correct stratified proportions."""
    with open(SPLITS_DIR / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(SPLITS_DIR / "val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(SPLITS_DIR / "test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    assert len(train_data) == 315  # 70% of 450
    assert len(val_data) == 67     # 15% of 450
    assert len(test_data) == 68    # 15% of 450
    assert len(train_data) + len(val_data) + len(test_data) == 450
