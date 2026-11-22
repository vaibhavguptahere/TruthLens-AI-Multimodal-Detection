"""
TruthLens AI - Image Preprocessing Pipeline
Standardizes, validates, and transforms images for CNN/Vision models (ResNet/ViT).
"""

from pathlib import Path
from typing import Union, Tuple
from PIL import Image, ImageOps
import torch
from torchvision import transforms


class ImagePreprocessor:
    """
    Standard image transformations strictly adhering to ImageNet distributions
    and defensive loading against corrupted or malformed files.
    """
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        
        # Standard ImageNet normalization coefficients
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Evaluation & Inference transform
        self.eval_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # Training transform with subtle data augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def load_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Safely loads an image, validates its format, and converts to 3-channel RGB.
        Raises ValueError if image file is corrupt or unreadable.
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at path: {path}")
            try:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    return img.convert("RGB")
            except Exception as e:
                raise ValueError(f"Corrupt or unsupported image file at {path}: {str(e)}")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def preprocess(self, image_input: Union[str, Path, Image.Image], is_training: bool = False) -> torch.Tensor:
        """
        Loads and transforms image into normalized (C, H, W) PyTorch Tensor.
        """
        img = self.load_image(image_input)
        transform = self.train_transform if is_training else self.eval_transform
        return transform(img)
