"""
Tests for Phase 3: Text and Image Preprocessing Pipelines & PyTorch DataLoaders.
"""

from pathlib import Path
import pytest
import torch
from PIL import Image

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.data.torch_dataset import MultimodalDataset, create_multimodal_dataloader, LABEL_MAP


def test_text_cleaning_and_tokenization():
    """Verify text preprocessor cleans whitespace and produces valid DistilBERT tokens."""
    preprocessor = TextPreprocessor(max_length=64)
    raw_text = "  Breaking \t news: \n Manipulated   image detected!   "
    cleaned = preprocessor.clean_text(raw_text)
    assert cleaned == "Breaking news: Manipulated image detected!"

    tokenized = preprocessor.tokenize(cleaned)
    assert "input_ids" in tokenized
    assert "attention_mask" in tokenized
    assert tokenized["input_ids"].shape == (1, 64)
    assert tokenized["attention_mask"].shape == (1, 64)
    # Token 101 is [CLS], 102 is [SEP] in BERT/DistilBERT
    assert tokenized["input_ids"][0][0].item() == 101


def test_image_preprocessing_transforms(tmp_path):
    """Verify image preprocessor loads, converts to RGB, and normalizes into (3, 224, 224) tensor."""
    img_path = tmp_path / "test_img.png"
    # Create test RGBA image
    test_img = Image.new("RGBA", (300, 400), color=(100, 150, 200, 255))
    test_img.save(img_path)

    preprocessor = ImagePreprocessor(image_size=224)
    img_tensor = preprocessor.preprocess(img_path, is_training=False)

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    # Check that tensor is normalized (not raw 0-255 uint8)
    assert img_tensor.dtype == torch.float32


def test_image_preprocessing_corrupt_file(tmp_path):
    """Verify that corrupt or non-image files raise ValueError as expected."""
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"THIS_IS_NOT_A_VALID_IMAGE_HEADER_123456789")

    preprocessor = ImagePreprocessor()
    with pytest.raises(ValueError, match="Corrupt or unsupported image"):
        preprocessor.preprocess(corrupt_file)


def test_multimodal_dataset_and_dataloader():
    """Verify that MultimodalDataset batches yield paired image and text tensors."""
    split_file = Path("data/splits/val.json")
    assert split_file.exists(), "val.json split file missing"

    dataset = MultimodalDataset(split_file=split_file, is_training=False)
    assert len(dataset) == 67

    first_item = dataset[0]
    assert "image" in first_item
    assert "input_ids" in first_item
    assert "attention_mask" in first_item
    assert "label" in first_item
    assert first_item["image"].shape == (3, 224, 224)
    assert first_item["input_ids"].shape == (128,)
    assert first_item["label"].item() in [0, 1, 2]

    # DataLoader batch test
    loader = create_multimodal_dataloader(split_file=split_file, batch_size=8, is_training=False)
    batch = next(iter(loader))
    assert batch["image"].shape == (8, 3, 224, 224)
    assert batch["input_ids"].shape == (8, 128)
    assert batch["attention_mask"].shape == (8, 128)
    assert batch["label"].shape == (8,)
