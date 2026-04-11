"""
evaluate.py — Evaluate the trained CandlestickViT model on the test set.

Generates accuracy, precision, recall, F1 scores, and confusion matrix.

Usage:
    python -m src.model.evaluate
    python src/model/evaluate.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from tqdm import tqdm

from config import (
    LABELS_DIR,
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    BATCH_SIZE,
)
from src.model.vit_model import load_model
from src.model.train import CandlestickDataset, get_transforms


def evaluate_model(
    csv_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device_str: str | None = None,
    save_results: bool = True,
) -> dict:
    """
    Evaluate the trained model on the test set.

    Args:
        csv_path: Path to labeled dataset CSV.
        checkpoint_path: Path to model checkpoint.
        device_str: Device string ('cuda', 'cpu'). Auto-detects if None.
        save_results: Whether to save results to files.

    Returns:
        Dictionary with evaluation metrics.
    """
    csv_path = csv_path or (LABELS_DIR / "labeled_dataset.csv")

    # Device
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"🖥️  Using device: {device}")

    # Load model
    print("\n📦 Loading model...")
    model = load_model(checkpoint_path, device=str(device))

    # Load test dataset
    print("\n📂 Loading test dataset...")
    test_dataset = CandlestickDataset(csv_path, "test", get_transforms("test"))
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Run inference
    print("\n🔍 Running evaluation...")
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    report_str = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*50}")
    print(f"📊 Evaluation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 (macro): {f1_macro:.4f}")
    print(f"\n{report_str}")
    print(f"{'='*50}")

    results = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(all_labels),
    }

    if save_results:
        # Save metrics JSON
        metrics_path = CHECKPOINTS_DIR / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"   💾 Metrics saved to: {metrics_path}")

        # Plot and save confusion matrix
        plot_confusion_matrix(cm, CHECKPOINTS_DIR / "confusion_matrix.png")

    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path | str,
) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        output_path: Path to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix — CandlestickViT", fontsize=14, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Confusion matrix saved to: {output_path}")


if __name__ == "__main__":
    evaluate_model()
