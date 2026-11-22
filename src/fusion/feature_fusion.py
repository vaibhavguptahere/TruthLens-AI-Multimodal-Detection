"""
TruthLens AI - Feature-Level Multimodal Fusion with Anti-Overfitting Architecture
- Independent projection bottlenecks for text and image into 256 dimensions
- LayerNorm + Dropout on projections
- Concatenated cross-modal representation (512-dim)
- Compact classifier with Early Stopping & L2 regularization
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

from src.models.text_baseline import TextBaselineModel
from src.models.image_baseline import ImageBaselineModel


class FeatureFusionModel(nn.Module):
    def __init__(
        self,
        text_backbone: Optional[TextBaselineModel] = None,
        image_backbone: Optional[ImageBaselineModel] = None,
        projection_dim: int = 256,
        num_classes: int = 3,
        num_heads: int = 4,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.text_model = text_backbone or TextBaselineModel()
        self.image_model = image_backbone or ImageBaselineModel(pretrained=False)

        text_dim = self.text_model.hidden_dim     # 768
        image_dim = self.image_model.feature_dim  # 2048
        self.projection_dim = projection_dim

        # Text projection with normalization
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        # Image projection with normalization
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        # Bidirectional Multi-Head Cross-Attention (Text queries Image, Image queries Text)
        self.cross_attn_text_to_img = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        self.cross_attn_img_to_text = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True
        )

        self.norm_cross_text = nn.LayerNorm(projection_dim)
        self.norm_cross_img = nn.LayerNorm(projection_dim)

        # Joint Multimodal Classifier
        # fused_dim = proj_text (256) + proj_image (256) + attn_text (256) + attn_img (256) + abs_diff (256) + mult (256) + cos_sim (1) = 1537
        fused_dim = (projection_dim * 6) + 1
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(384, 96),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Dropout(dropout_prob * 0.5),
            nn.Linear(96, num_classes)
        )

    def extract_multimodal_embeddings(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_text_emb = self.text_model.extract_features(input_ids, attention_mask)
        raw_img_emb = self.image_model.extract_features(images)

        proj_text = self.text_proj(raw_text_emb)   # (B, 256)
        proj_image = self.image_proj(raw_img_emb) # (B, 256)

        # Reshape to sequence length 1 for multi-head cross-attention: (B, 1, 256)
        text_seq = proj_text.unsqueeze(1)
        img_seq = proj_image.unsqueeze(1)

        # Text attends to Image
        attn_text_out, _ = self.cross_attn_text_to_img(query=text_seq, key=img_seq, value=img_seq)
        attn_text = self.norm_cross_text(proj_text + attn_text_out.squeeze(1))

        # Image attends to Text
        attn_img_out, _ = self.cross_attn_img_to_text(query=img_seq, key=text_seq, value=text_seq)
        attn_img = self.norm_cross_img(proj_image + attn_img_out.squeeze(1))

        # Interaction cues
        diff = torch.abs(proj_text - proj_image)
        prod = proj_text * proj_image
        
        # Normalized Cosine Alignment
        norm_text = nn.functional.normalize(proj_text, p=2, dim=-1)
        norm_img = nn.functional.normalize(proj_image, p=2, dim=-1)
        cos_sim = torch.sum(norm_text * norm_img, dim=-1, keepdim=True)

        # Fused multi-head cross-attentive representation (1537-dim)
        fused_repr = torch.cat([proj_text, proj_image, attn_text, attn_img, diff, prod, cos_sim], dim=-1)

        return proj_text, proj_image, fused_repr

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        proj_text, proj_image, fused_repr = self.extract_multimodal_embeddings(
            images, input_ids, attention_mask
        )
        logits = self.classifier(fused_repr)

        if return_embeddings:
            return logits, {
                "proj_text": proj_text,
                "proj_image": proj_image,
                "fused_repr": fused_repr
            }
        return logits, None

    def predict_proba(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(images, input_ids, attention_mask)
            return torch.softmax(logits, dim=-1)
