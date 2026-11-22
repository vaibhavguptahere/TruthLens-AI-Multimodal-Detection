"""
TruthLens AI - Decision-Level Multimodal Fusion Engine
Combines predictions from independently trained text and image models.
Implements calibrated probability aggregation, cross-modal conflict detection,
and automated routing to UNCERTAIN when signals diverge.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np


class DecisionFusionEngine:
    """
    Late-stage decision fusion module.
    Aggregates image and text prediction probabilities with dynamic conflict resolution:
    - If both models agree on REAL or FAKE with high confidence -> Consistent Decision
    - If models produce conflicting predictions (e.g., Image says FAKE, Text says REAL) -> Boost UNCERTAIN
    - If overall fused confidence falls below a calibrated threshold -> Output UNCERTAIN
    """
    def __init__(
        self,
        text_weight: float = 0.50,
        image_weight: float = 0.50,
        conflict_threshold: float = 0.40,
        min_confidence: float = 0.60
    ):
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.conflict_threshold = conflict_threshold
        self.min_confidence = min_confidence
        self.class_names = ["REAL", "FAKE", "UNCERTAIN"]

    def fuse_probabilities(
        self,
        text_probs: torch.Tensor,
        image_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Fuses probabilities for a batch of predictions.
        Args:
            text_probs: (B, 3) tensor representing [REAL, FAKE, UNCERTAIN] from text model
            image_probs: (B, 3) tensor representing [REAL, FAKE, UNCERTAIN] from image model
        Returns:
            fused_probs: (B, 3) calibrated fused probability distribution
            predicted_classes: (B,) predicted class indices
            diagnostics: Dict containing conflict scores and metadata
        """
        # Ensure tensors are float on CPU
        p_txt = text_probs.detach().cpu()
        p_img = image_probs.detach().cpu()

        batch_size = p_txt.shape[0]
        fused_list = []
        conflict_scores = []

        for i in range(batch_size):
            t = p_txt[i]
            im = p_img[i]

            # 1. Base weighted sum
            base_fused = (self.text_weight * t) + (self.image_weight * im)

            # 2. Compute cross-modal divergence / conflict
            # High conflict occurs when one modal strongly favors REAL (idx 0) while other favors FAKE (idx 1)
            polar_conflict_1 = abs(t[0].item() - im[0].item())
            polar_conflict_2 = abs(t[1].item() - im[1].item())
            conflict_score = max(polar_conflict_1, polar_conflict_2)
            conflict_scores.append(conflict_score)

            # 3. Dynamic Uncertainty Adjustment:
            # If conflict is higher than threshold, reallocate probability mass to UNCERTAIN (idx 2)
            if conflict_score > self.conflict_threshold:
                penalty = conflict_score * 0.5
                adjusted_fused = base_fused.clone()
                # Subtract penalty from REAL and FAKE, add to UNCERTAIN
                adjusted_fused[0] = max(0.0, adjusted_fused[0].item() - penalty / 2)
                adjusted_fused[1] = max(0.0, adjusted_fused[1].item() - penalty / 2)
                adjusted_fused[2] = adjusted_fused[2].item() + penalty
                # Renormalize
                adjusted_fused = adjusted_fused / adjusted_fused.sum()
            else:
                adjusted_fused = base_fused

            # 4. Low-confidence fallback to UNCERTAIN
            max_prob, top_class = torch.max(adjusted_fused, dim=-1)
            if max_prob.item() < self.min_confidence and top_class.item() != 2:
                # If peak confidence is too weak, enforce UNCERTAIN
                adjusted_fused[2] += (self.min_confidence - max_prob.item())
                adjusted_fused = adjusted_fused / adjusted_fused.sum()

            fused_list.append(adjusted_fused)

        fused_tensor = torch.stack(fused_list, dim=0)
        predicted_classes = torch.argmax(fused_tensor, dim=-1)

        diagnostics = {
            "conflict_scores": conflict_scores,
            "mean_conflict": float(np.mean(conflict_scores)),
            "text_weight": self.text_weight,
            "image_weight": self.image_weight
        }
        return fused_tensor, predicted_classes, diagnostics
