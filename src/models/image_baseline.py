"""
TruthLens AI - Image-Only Baseline with Anti-Overfitting Architecture
- Pretrained ResNet-50 visual feature extractor
- Bottleneck compression to 512 dimensions
- Batch Normalization + Weight Decay + Dropout (p=0.4)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models


class ImageBaselineModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout_prob: float = 0.4
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Retain convolutional stem up to layer4 for feature extraction & Grad-CAM
        self.conv_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.avgpool = resnet.avgpool
        self.feature_dim = 2048

        # Anti-overfitting classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_prob * 0.75),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(images)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.extract_features(images)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits, None

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(images)
            return torch.softmax(logits, dim=-1)
