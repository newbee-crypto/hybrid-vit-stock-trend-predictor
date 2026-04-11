"""
attention_viz.py — Attention visualization and description for ViT.

Extracts attention weights from the ViT model and generates natural
language descriptions of which chart regions the model focused on.

Usage:
    from src.xai.attention_viz import describe_attention
    description = describe_attention(model, image_path)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, CLASS_NAMES


# Inference transform
ATTENTION_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Region mapping for 14x14 grid → chart regions
REGION_MAP = {
    "top_left": {"rows": (0, 4), "cols": (0, 4), "desc": "opening candles (upper wicks)"},
    "top_center": {"rows": (0, 4), "cols": (5, 9), "desc": "middle candles (upper wicks)"},
    "top_right": {"rows": (0, 4), "cols": (10, 13), "desc": "recent candles (upper wicks)"},
    "mid_left": {"rows": (5, 9), "cols": (0, 4), "desc": "opening candles (bodies)"},
    "mid_center": {"rows": (5, 9), "cols": (5, 9), "desc": "middle candles (bodies)"},
    "mid_right": {"rows": (5, 9), "cols": (10, 13), "desc": "recent candles (bodies)"},
    "bottom_left": {"rows": (10, 13), "cols": (0, 4), "desc": "opening candles (volume/lower wicks)"},
    "bottom_center": {"rows": (10, 13), "cols": (5, 9), "desc": "middle candles (volume/lower wicks)"},
    "bottom_right": {"rows": (10, 13), "cols": (10, 13), "desc": "recent candles (volume/lower wicks)"},
}


def extract_attention_map(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Extract the averaged attention map from the last transformer block.

    Args:
        model: CandlestickViT model.
        image_tensor: Preprocessed image tensor (1, 3, 224, 224).
        device: Computation device.

    Returns:
        Attention map array of shape (14, 14) normalized to [0, 1].
    """
    image_tensor = image_tensor.to(device)
    model.eval()

    # Hook to capture attention weights
    attention_weights = []

    def attn_hook(module, input, output):
        """Capture attention output."""
        attention_weights.append(output.detach())

    # Register hook on the last block's attention dropout
    hook = model.backbone.blocks[-1].attn.attn_drop.register_forward_hook(attn_hook)

    with torch.no_grad():
        _ = model(image_tensor)

    hook.remove()

    if not attention_weights:
        # Fallback: return uniform attention
        return np.ones((14, 14)) / (14 * 14)

    # Process attention: shape is (batch, heads, tokens, tokens)
    attn = attention_weights[0]  # (1, heads, N+1, N+1)

    # Average over heads
    attn = attn.mean(dim=1)  # (1, N+1, N+1)

    # Get CLS token attention to all patches (skip CLS-to-CLS)
    cls_attn = attn[0, 0, 1:]  # (N,)

    # Reshape to spatial grid
    num_patches_side = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(num_patches_side, num_patches_side)
    attn_map = attn_map.cpu().numpy()

    # Normalize
    attn_min, attn_max = attn_map.min(), attn_map.max()
    if attn_max - attn_min > 0:
        attn_map = (attn_map - attn_min) / (attn_max - attn_min)
    else:
        attn_map = np.ones_like(attn_map) / attn_map.size

    return attn_map


def identify_top_regions(
    attention_map: np.ndarray,
    top_k: int = 3,
) -> list[dict]:
    """
    Identify the top-K regions the model focused on.

    Args:
        attention_map: Attention map of shape (14, 14).
        top_k: Number of top regions to return.

    Returns:
        List of dictionaries with region info:
            - name: Region name (e.g., 'mid_right')
            - description: Human-readable description
            - attention_score: Average attention in that region
    """
    region_scores = []

    for region_name, region_info in REGION_MAP.items():
        row_start, row_end = region_info["rows"]
        col_start, col_end = region_info["cols"]

        # Clip to valid range
        row_end = min(row_end, attention_map.shape[0] - 1)
        col_end = min(col_end, attention_map.shape[1] - 1)

        region_attn = attention_map[row_start:row_end + 1, col_start:col_end + 1]
        avg_score = float(region_attn.mean())

        region_scores.append({
            "name": region_name,
            "description": region_info["desc"],
            "attention_score": avg_score,
        })

    # Sort by attention score descending
    region_scores.sort(key=lambda x: x["attention_score"], reverse=True)

    return region_scores[:top_k]


def generate_attention_description(
    top_regions: list[dict],
    prediction: str,
    confidence: float,
) -> str:
    """
    Generate a natural language description of the model's attention pattern.

    Args:
        top_regions: List of top attention regions from identify_top_regions().
        prediction: Predicted class name ('Up', 'Down', 'Neutral').
        confidence: Prediction confidence (0-1).

    Returns:
        Human-readable description string.
    """
    if not top_regions:
        return "The model's attention pattern could not be determined."

    # Build description
    descriptions = []
    for i, region in enumerate(top_regions):
        score_pct = region["attention_score"] * 100
        if i == 0:
            descriptions.append(
                f"The model primarily focused on the {region['description']} "
                f"(attention: {score_pct:.1f}%)"
            )
        else:
            descriptions.append(
                f"followed by the {region['description']} "
                f"(attention: {score_pct:.1f}%)"
            )

    # Combine descriptions
    attention_text = ", ".join(descriptions) + "."

    # Add interpretation based on prediction
    interpretation = _get_interpretation(top_regions, prediction)

    full_description = f"{attention_text} {interpretation}"
    return full_description


def _get_interpretation(
    top_regions: list[dict],
    prediction: str,
) -> str:
    """
    Generate an interpretation of why the model's attention pattern
    might lead to the given prediction.

    Args:
        top_regions: List of top attention regions.
        prediction: Predicted class name.

    Returns:
        Interpretation string.
    """
    primary_region = top_regions[0]["name"] if top_regions else ""

    interpretations = {
        "Up": {
            "top": "The focus on upper wicks suggests the model detected buying pressure pushing prices higher.",
            "mid": "The focus on candle bodies indicates the model identified consistent bullish closes.",
            "bottom": "The focus on volume/lower regions suggests strong buying volume supporting the uptrend.",
        },
        "Down": {
            "top": "The focus on upper wicks suggests the model detected rejection at higher levels.",
            "mid": "The focus on candle bodies indicates the model identified declining price action.",
            "bottom": "The focus on lower wicks/volume suggests selling pressure driving prices down.",
        },
        "Neutral": {
            "top": "The focus on upper regions suggests mixed signals with no clear directional bias.",
            "mid": "The focus on candle bodies suggests sideways consolidation in the price action.",
            "bottom": "The focus on volume patterns suggests indecisive market participants.",
        },
    }

    # Determine vertical position
    if "top" in primary_region:
        position = "top"
    elif "bottom" in primary_region:
        position = "bottom"
    else:
        position = "mid"

    return interpretations.get(prediction, {}).get(
        position,
        "The model's attention pattern suggests a complex analysis of multiple chart features."
    )


def describe_attention(
    image_input: str | Path | Image.Image,
    model: torch.nn.Module | None = None,
    device_str: str | None = None,
    top_k: int = 3,
) -> dict:
    """
    Full attention analysis pipeline for a candlestick chart image.

    Args:
        image_input: Path to image or PIL Image.
        model: Pre-loaded model.
        device_str: Device string.
        top_k: Number of top regions to identify.

    Returns:
        Dictionary with:
            - attention_map: 14x14 attention array
            - top_regions: Top-K regions with scores
            - description: Natural language description
            - prediction: Predicted class
            - confidence: Prediction confidence
    """
    from src.model.vit_model import load_model as _load_model

    if device_str is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_str

    if model is None:
        model = _load_model(device=device)

    # Load image
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    image_tensor = ATTENTION_TRANSFORM(image).unsqueeze(0)

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_class = int(probs.argmax().item())
        confidence = float(probs[pred_class].item())

    prediction = CLASS_NAMES[pred_class]

    # Extract attention map
    attention_map = extract_attention_map(model, image_tensor, device)

    # Identify top regions
    top_regions = identify_top_regions(attention_map, top_k)

    # Generate description
    description = generate_attention_description(top_regions, prediction, confidence)

    return {
        "attention_map": attention_map,
        "top_regions": top_regions,
        "description": description,
        "prediction": prediction,
        "confidence": confidence,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze model attention")
    parser.add_argument("--image", type=str, required=True, help="Path to chart image")
    args = parser.parse_args()

    result = describe_attention(args.image)

    print(f"\n{'='*60}")
    print(f"🔍 Attention Analysis:")
    print(f"   Prediction: {result['prediction']} ({result['confidence']:.2%})")
    print(f"\n   Top Regions:")
    for region in result["top_regions"]:
        print(f"     • {region['description']}: {region['attention_score']:.3f}")
    print(f"\n   Description:")
    print(f"     {result['description']}")
    print(f"{'='*60}")
