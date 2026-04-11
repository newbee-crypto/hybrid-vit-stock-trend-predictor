"""
gradcam.py — Grad-CAM visualization for the CandlestickViT model.

Generates heatmap overlays showing which regions of the candlestick chart
the model focused on when making its prediction.

Usage:
    from src.xai.gradcam import generate_gradcam
    overlay = generate_gradcam(model, image_path)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, CLASS_NAMES
from src.model.vit_model import load_model


# Inference transform (same as inference.py)
GRADCAM_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ViTGradCAM:
    """
    Grad-CAM implementation adapted for Vision Transformers.

    Instead of using convolutional feature maps, this implementation
    uses the output of the last transformer block's LayerNorm to
    generate spatial attention maps.

    Attributes:
        model: The CandlestickViT model.
        device: Computation device.
        activations: Stored forward activations.
        gradients: Stored backward gradients.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Initialize Grad-CAM for ViT.

        Args:
            model: CandlestickViT model instance.
            device: Device string ('cuda', 'cpu').
        """
        self.model = model
        self.device = device
        self.activations = None
        self.gradients = None

        # Register hooks on the last transformer block's norm layer
        target_layer = model.backbone.blocks[-1].norm1
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Store activations from the forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients from the backward pass."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224).
            target_class: Class index to generate CAM for. If None,
                uses the predicted class.

        Returns:
            Heatmap array of shape (14, 14) normalized to [0, 1].
        """
        image_tensor = image_tensor.to(self.device)
        self.model.eval()

        # Forward pass
        output = self.model(image_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward(retain_graph=True)

        # Get activations and gradients
        activations = self.activations  # (1, num_patches+1, dim)
        gradients = self.gradients      # (1, num_patches+1, dim)

        # Remove CLS token (first token)
        activations = activations[:, 1:, :]  # (1, num_patches, dim)
        gradients = gradients[:, 1:, :]      # (1, num_patches, dim)

        # Global average pooling of gradients → weights
        weights = gradients.mean(dim=-1, keepdim=True)  # (1, num_patches, 1)

        # Weighted combination
        cam = (activations * weights).sum(dim=-1)  # (1, num_patches)
        cam = F.relu(cam)  # Only positive contributions

        # Reshape to spatial grid (14x14 for patch_size=16 with 224 input)
        num_patches_side = int(cam.shape[1] ** 0.5)
        cam = cam.reshape(1, num_patches_side, num_patches_side)

        # Normalize to [0, 1]
        cam = cam.squeeze(0).cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def generate_gradcam(
    image_input: str | Path | Image.Image,
    model: torch.nn.Module | None = None,
    target_class: int | None = None,
    device_str: str | None = None,
    alpha: float = 0.5,
) -> dict:
    """
    Generate Grad-CAM heatmap and overlay for a candlestick chart image.

    Args:
        image_input: Path to image file or PIL Image.
        model: Pre-loaded model. If None, loads from default checkpoint.
        target_class: Class to visualize. If None, uses predicted class.
        device_str: Device string ('cuda', 'cpu').
        alpha: Overlay transparency (0=original, 1=heatmap only).

    Returns:
        Dictionary with:
            - heatmap: Raw heatmap array (14x14)
            - overlay: PIL Image with heatmap overlay (224x224)
            - prediction: Predicted class name
            - confidence: Prediction confidence
    """
    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    # Load model
    if model is None:
        model = load_model(device=device)

    # Load image
    if isinstance(image_input, (str, Path)):
        original_image = Image.open(image_input).convert("RGB")
    else:
        original_image = image_input.convert("RGB")

    original_image = original_image.resize(IMAGE_SIZE, Image.LANCZOS)

    # Prepare input tensor
    image_tensor = GRADCAM_TRANSFORM(original_image).unsqueeze(0)

    # Generate Grad-CAM
    gradcam = ViTGradCAM(model, device)
    heatmap = gradcam.generate(image_tensor, target_class)

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_class = int(probs.argmax().item())
        confidence = float(probs[pred_class].item())

    # Create overlay image
    overlay = create_overlay(original_image, heatmap, alpha)

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "prediction": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "probabilities": {
            name: float(probs[i].item())
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def create_overlay(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Create an overlay of the Grad-CAM heatmap on the original image.

    Args:
        original: Original PIL Image.
        heatmap: Heatmap array (H, W) normalized to [0, 1].
        alpha: Overlay transparency.

    Returns:
        PIL Image with heatmap overlay.
    """
    # Resize heatmap to original image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            original.size, Image.LANCZOS
        )
    ).astype(np.float32) / 255.0

    # Create colormap
    fig, ax = plt.subplots(figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    ax.imshow(np.array(original))
    ax.imshow(heatmap_resized, cmap="jet", alpha=alpha)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Convert matplotlib figure to PIL image
    fig.canvas.draw()
    overlay_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    overlay_array = overlay_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    overlay = Image.fromarray(overlay_array).resize(IMAGE_SIZE, Image.LANCZOS)
    return overlay


def save_gradcam(
    image_input: str | Path,
    output_path: str | Path,
    model: torch.nn.Module | None = None,
    device_str: str | None = None,
) -> dict:
    """
    Generate and save Grad-CAM visualization for an image.

    Args:
        image_input: Path to input candlestick chart image.
        output_path: Path to save the overlay image.
        model: Pre-loaded model.
        device_str: Device string.

    Returns:
        Prediction result dictionary.
    """
    result = generate_gradcam(image_input, model=model, device_str=device_str)
    result["overlay"].save(output_path)
    print(f"✅ Grad-CAM overlay saved to: {output_path}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap")
    parser.add_argument("--image", type=str, required=True, help="Path to chart image")
    parser.add_argument("--output", type=str, default="gradcam_overlay.png", help="Output path")
    args = parser.parse_args()

    result = save_gradcam(args.image, args.output)
    print(f"   Prediction: {result['prediction']} ({result['confidence']:.2%})")
