"""
TruthLens AI - Visual Explainability (Grad-CAM)
Generates spatial Class Activation Mapping (Grad-CAM) heatmaps from CNN/ResNet convolutional layers
to highlight visual manipulation artifacts or authentic regions.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAMExplainer:
    """
    Grad-CAM implementation for ResNet-50 visual feature extractor.
    Hooks into layer4 (final convolutional block) to compute spatial gradient weights.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # If model is ImageBaselineModel, target the final layer4 in conv_stem
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "conv_stem"):
            # ResNet conv_stem: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4 (index 7)
            self.target_layer = model.conv_stem[7]
        elif hasattr(model, "image_model") and hasattr(model.image_model, "conv_stem"):
            self.target_layer = model.image_model.conv_stem[7]
        else:
            raise ValueError("Unable to automatically identify target conv layer for Grad-CAM.")

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes normalized (224, 224) 2D Grad-CAM heatmap array in range [0, 1].
        Args:
            image_tensor: (1, 3, 224, 224) normalized input tensor
            target_class: Target class index (0: REAL, 1: FAKE, 2: UNCERTAIN). Defaults to argmax logit.
        """
        self.model.zero_grad()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Forward pass
        logits, _ = self.model(image_tensor)
        if target_class is None:
            target_class = torch.argmax(logits, dim=-1).item()

        # Backward pass on target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Global average pooling on gradients: (C,)
        pooled_gradients = torch.mean(self.gradients[0], dim=[1, 2])

        # Weight the activation maps: (C, H, W)
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]

        # Channel-wise summation + ReLU to isolate positive influences
        heatmap = torch.sum(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Normalize heatmap to [0, 1]
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        # Resize heatmap to match original image size (224, 224)
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_resized = heatmap_img.resize((224, 224), resample=Image.BILINEAR)
        return np.array(heatmap_resized) / 255.0

    def cleanup(self):
        """Removes PyTorch hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
