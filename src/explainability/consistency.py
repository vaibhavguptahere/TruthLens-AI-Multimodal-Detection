"""
TruthLens AI - Cross-Modal Semantic Consistency Analyzer
Measures directional semantic alignment between visual embeddings and textual embeddings.
Evaluates out-of-context miscaptioning without claiming similarity proves image authenticity.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np


class SemanticConsistencyAnalyzer:
    """
    Computes cross-modal cosine similarity between projected text and image representations.
    Strictly adheres to AGENTS.md rule: "Never claim CLIP/cosine similarity proves image authenticity."
    Instead, it flags:
    - High Consistency: Image and Text corroborate each other.
    - Low Consistency (Mismatch): Authentic image paired with fabricated/unrelated caption.
    - Ambiguous: Divergent or low-confidence alignment.
    """
    def __init__(self, aligned_threshold: float = 0.55, misaligned_threshold: float = 0.25):
        self.aligned_threshold = aligned_threshold
        self.misaligned_threshold = misaligned_threshold

    def compute_consistency(
        self,
        proj_text: torch.Tensor,
        proj_image: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Args:
            proj_text: (B, D) projected text embeddings
            proj_image: (B, D) projected image embeddings
        Returns:
            Dict containing cosine_similarity, consistency_status, and interpretation.
        """
        # Normalize vectors to unit sphere
        u_text = F.normalize(proj_text, p=2, dim=-1)
        u_img = F.normalize(proj_image, p=2, dim=-1)

        # Batch cosine similarity
        cosine_sim = torch.sum(u_text * u_img, dim=-1).detach().cpu().numpy()
        sim_val = float(cosine_sim[0]) if cosine_sim.ndim > 0 else float(cosine_sim)

        if sim_val >= self.aligned_threshold:
            status = "ALIGNED"
            interpretation = "Visual context and textual claims exhibit high semantic correspondence."
        elif sim_val <= self.misaligned_threshold:
            status = "MISALIGNED"
            interpretation = "Potential out-of-context miscaptioning: Caption claim conflicts with visual content."
        else:
            status = "AMBIGUOUS"
            interpretation = "Cross-modal correspondence is weak or partially contradictory."

        return {
            "cosine_similarity": float(np.round(sim_val, 4)),
            "consistency_status": status,
            "interpretation": interpretation
        }
