"""
TruthLens AI - Text-Only Baseline with Anti-Overfitting Architecture
- DistilBERT-base feature extractor
- High dropout (p=0.4)
- LayerNorm regularization
- Bottleneck MLP with residual skip
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class TextBaselineModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "distilbert-base-uncased",
        num_classes: int = 3,
        dropout_prob: float = 0.4
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        self.config = AutoConfig.from_pretrained(backbone_name)
        self.transformer = AutoModel.from_pretrained(backbone_name, config=self.config)
        self.hidden_dim = self.config.hidden_size  # 768

        # Anti-overfitting classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_prob * 0.75),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Pooled [CLS] representation
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.extract_features(input_ids, attention_mask)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits, None

    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids, attention_mask)
            return torch.softmax(logits, dim=-1)
