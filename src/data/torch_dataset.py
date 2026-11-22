"""
TruthLens AI - PyTorch Multimodal Dataset & DataLoader Factory
Provides unified loading of paired images and texts with standard tensor collation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.image_preprocessor import ImagePreprocessor

# Deterministic label mapping
LABEL_MAP = {
    "REAL": 0,
    "FAKE": 1,
    "UNCERTAIN": 2
}
REV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class MultimodalDataset(Dataset):
    """
    PyTorch Dataset yielding synchronized (image_tensor, input_ids, attention_mask, label_idx, metadata).
    """
    def __init__(
        self,
        split_file: Union[str, Path],
        data_root: Union[str, Path] = Path("."),
        image_preprocessor: Optional[ImagePreprocessor] = None,
        text_preprocessor: Optional[TextPreprocessor] = None,
        is_training: bool = False
    ):
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.is_training = is_training

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.samples: List[Dict[str, Any]] = json.load(f)

        self.img_proc = image_preprocessor or ImagePreprocessor()
        self.txt_proc = text_preprocessor or TextPreprocessor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        # 1. Process image
        rel_img_path = item["image_rel_path"]
        abs_img_path = self.data_root / rel_img_path
        image_tensor = self.img_proc.preprocess(abs_img_path, is_training=self.is_training)

        # 2. Process text
        tokenized = self.txt_proc.tokenize(item["text"], return_tensors="pt")
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        # 3. Target label
        label_str = item["pair_label"]
        label_idx = LABEL_MAP.get(label_str, 2)  # Default UNCERTAIN if unrecognized

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "sample_id": item["sample_id"],
            "manipulation_type": item.get("manipulation_type", "unknown"),
            "image_authenticity": item.get("image_authenticity", "unknown"),
            "text_claim_status": item.get("text_claim_status", "unknown"),
            "image_text_consistency": item.get("image_text_consistency", "unknown")
        }


def create_multimodal_dataloader(
    split_file: Union[str, Path],
    batch_size: int = 16,
    is_training: bool = False,
    shuffle: Optional[bool] = None,
    num_workers: int = 0
) -> DataLoader:
    """Factory helper to construct DataLoader with correct batch collation."""
    dataset = MultimodalDataset(split_file=split_file, is_training=is_training)
    do_shuffle = shuffle if shuffle is not None else is_training
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        num_workers=num_workers
    )
