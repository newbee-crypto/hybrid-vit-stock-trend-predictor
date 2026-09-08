"""
inference.py — Single-image inference for the CandlestickViT model.

Loads a trained checkpoint and predicts trend direction from a candlestick
chart image. Includes fallback mode if no checkpoint is available.

Usage:
    python -m src.model.inference --image path/to/chart.png
    python src/model/inference.py --image path/to/chart.png

    from src.model.inference import predict_image
    result = predict_image("path/to/chart.png")
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, CLASS_NAMES, CHECKPOINTS_DIR, BEST_CHECKPOINT
from src.model.vit_model import load_model


# Standard ImageNet normalization for inference
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(
    image_input: str | Path | Image.Image,
    checkpoint_path: str | Path | None = None,
    device_str: str | None = None,
    model: torch.nn.Module | None = None,
    aux_features: list[float] | tuple[float, float, float] | np.ndarray | None = None,
) -> dict:
    """
    Predict the trend direction from a single candlestick chart image.

    Args:
        image_input: Path to an image file or a PIL Image object.
        checkpoint_path: Path to the model checkpoint. If None, uses default.
        device_str: Device string ('cuda', 'cpu'). Auto-detects if None.
        model: Pre-loaded model. If provided, skips loading from checkpoint.
        aux_features: Optional RSI/MACD/trend_score values for hybrid models.

    Returns:
        Dictionary with:
            - prediction: str ('Up', 'Down', 'Neutral')
            - confidence: float (0-1)
            - probabilities: dict mapping class names to probabilities
            - class_id: int (predicted class index)
            - is_fallback: bool (True if using untrained model)
    """
    # Device
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load model if not provided
    if model is None:
        model = load_model(checkpoint_path, device=str(device))

    # Load image
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input.convert("RGB")

    # Transform
    image_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)
    aux_tensor = None
    if getattr(model, "use_aux_features", False):
        if aux_features is None:
            aux_tensor = torch.zeros((1, model.num_aux_features), device=device)
        else:
            aux_array = np.asarray(aux_features, dtype=np.float32).reshape(1, -1)
            aux_tensor = torch.from_numpy(aux_array).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor, aux_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    # Extract results
    class_id = int(probabilities.argmax().item())
    confidence = float(probabilities[class_id].item())
    probs_dict = {
        name: float(probabilities[i].item())
        for i, name in enumerate(CLASS_NAMES)
    }

    # Check if this is a fallback prediction (no checkpoint used)
    if checkpoint_path is None:
        checkpoint_candidates = [
            CHECKPOINTS_DIR / BEST_CHECKPOINT,
            CHECKPOINTS_DIR / "best_model.pth",
            CHECKPOINTS_DIR / "bestmodel.pth",
        ]
        resolved_checkpoint_path = next(
            (path for path in checkpoint_candidates if path.exists()),
            checkpoint_candidates[0],
        )
    else:
        resolved_checkpoint_path = Path(checkpoint_path)
    is_fallback = not resolved_checkpoint_path.exists()

    result = {
        "prediction": CLASS_NAMES[class_id],
        "confidence": confidence,
        "probabilities": probs_dict,
        "class_id": class_id,
        "is_fallback": is_fallback,
    }

    return result


def build_live_aux_features(
    rsi: float | None = None,
    macd: float | None = None,
) -> list[float] | None:
    """Build live inference features using only RSI and MACD."""
    if rsi is None or macd is None:
        return None
    return [float(rsi), float(macd), 0.0]


def predict_batch(
    image_paths: list[str | Path],
    checkpoint_path: str | Path | None = None,
    device_str: str | None = None,
    aux_features_list: list[list[float] | tuple[float, float, float] | np.ndarray] | None = None,
) -> list[dict]:
    """
    Predict trend direction for a batch of candlestick chart images.

    Args:
        image_paths: List of paths to image files.
        checkpoint_path: Path to the model checkpoint.
        device_str: Device string ('cuda', 'cpu').

    Returns:
        List of prediction dictionaries.
    """
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load model once
    model = load_model(checkpoint_path, device=str(device))

    results = []
    for idx, path in enumerate(image_paths):
        aux_features = None if aux_features_list is None else aux_features_list[idx]
        result = predict_image(
            path,
            model=model,
            device_str=str(device),
            aux_features=aux_features,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict stock trend from candlestick chart")
    parser.add_argument("--image", type=str, required=True, help="Path to candlestick chart image")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    result = predict_image(args.image, device_str=args.device)

    print(f"\n{'='*50}")
    print("Prediction Result:")
    print(f"   Trend: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"\n   Probabilities:")
    for name, prob in result['probabilities'].items():
        bar = "#" * int(prob * 30)
        print(f"     {name:>8}: {prob:.4f} {bar}")
    if result['is_fallback']:
        print("\n   WARNING: Using untrained model (fallback mode)")
    print(f"{'='*50}")
