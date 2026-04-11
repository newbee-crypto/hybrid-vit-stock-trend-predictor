"""
train.py — Training script for the CandlestickViT model.

Supports both local GPU training and Google Colab execution.
Uses AdamW optimizer with CosineAnnealingLR scheduler and early stopping.

Usage:
    python -m src.model.train
    python src/model/train.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from config import (
    LABELS_DIR,
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    BEST_CHECKPOINT,
    CLASS_NAMES,
    NUM_CLASSES,
)
from src.model.vit_model import CandlestickViT

HEAD_ONLY_WARMUP_EPOCHS = 3
LABEL_SMOOTHING = 0.05


class CandlestickDataset(Dataset):
    """
    PyTorch Dataset for candlestick chart images.

    Loads images and labels from the labeled dataset CSV file.

    Attributes:
        data: DataFrame with image paths and labels.
        transform: Image transformations to apply.
    """

    def __init__(
        self,
        csv_path: Path | str,
        split: str = "train",
        transform: transforms.Compose | None = None,
        include_aux_features: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to the labeled dataset CSV.
            split: Data split to use ('train', 'val', 'test').
            transform: Optional image transformations.
        """
        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.include_aux_features = include_aux_features

        print(f"   Loaded {split} set: {len(self.data)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_id).
        """
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label_id"])

        # Load and convert image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.include_aux_features:
            aux_features = torch.tensor(
                [
                    float(row.get("RSI", 0.0) or 0.0),
                    float(row.get("MACD", 0.0) or 0.0),
                    float(row.get("trend_score", 0.0) or 0.0),
                ],
                dtype=torch.float32,
            )
            return image, label, aux_features

        return image, label

    def label_counts(self) -> pd.Series:
        """Return class counts for the active split."""
        return self.data["label_id"].value_counts().sort_index()


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Get image transformations for the specified split.

    Training includes augmentations; validation/test only normalize.

    Args:
        split: Data split ('train', 'val', 'test').

    Returns:
        Composed transforms.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.97, 1.03),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Validate the model on the validation set.

    Args:
        model: The model to validate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    csv_path: Path | str | None = None,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    device_str: str | None = None,
) -> dict:
    """
    Full training loop for the CandlestickViT model.

    Args:
        csv_path: Path to labeled dataset CSV. Defaults to config path.
        num_epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        device_str: Device string ('cuda', 'cpu'). Auto-detects if None.

    Returns:
        Dictionary with training history and best metrics.
    """
    csv_path = csv_path or (LABELS_DIR / "labeled_dataset.csv")

    # Device setup
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    print("\nLoading datasets...")
    train_dataset = CandlestickDataset(csv_path, "train", get_transforms("train"))
    val_dataset = CandlestickDataset(csv_path, "val", get_transforms("val"))

    class_counts = train_dataset.label_counts().reindex(range(NUM_CLASSES), fill_value=0)
    class_weights = len(train_dataset) / (NUM_CLASSES * class_counts.clip(lower=1))
    sample_weights = train_dataset.data["label_id"].map(class_weights).astype(float).to_numpy()
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    print(f"   Class counts: {class_counts.to_dict()}")
    print(
        "   Class weights: "
        f"{ {int(k): round(float(v), 4) for k, v in class_weights.items()} }"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Model setup
    print("\nBuilding model...")
    model = CandlestickViT(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    for param in model.backbone.parameters():
        param.requires_grad = False
    print(f"   Frozen ViT backbone for first {HEAD_ONLY_WARMUP_EPOCHS} epochs")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights.values, dtype=torch.float32, device=device),
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        if epoch == HEAD_ONLY_WARMUP_EPOCHS + 1:
            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": learning_rate * 0.2},
                    {"params": model.classifier.parameters(), "lr": learning_rate},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            )
            print("   Unfroze ViT backbone for full fine-tuning")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(
            f"Epoch [{epoch:3d}/{num_epochs}] "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"| LR: {current_lr:.6f} "
            f"| Time: {epoch_time:.1f}s"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }
            checkpoint_path = CHECKPOINTS_DIR / BEST_CHECKPOINT
            torch.save(checkpoint, checkpoint_path)
            print(f"   Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
                break

    total_time = time.time() - start_time

    # Save training history
    history_path = CHECKPOINTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Best val accuracy: {best_val_acc:.4f}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   Checkpoint: {CHECKPOINTS_DIR / BEST_CHECKPOINT}")
    print(f"   History: {history_path}")
    print(f"{'='*60}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "total_time": total_time,
        "history": history,
    }


if __name__ == "__main__":
    train_model()
