"""
vit_model.py -- Vision Transformer model for candlestick chart classification.

Supports both the current project checkpoint format and an older Kaggle/Colab
checkpoint layout that used:
- `vit.*` backbone parameter names
- `fc.*` classifier parameter names
- 3 auxiliary technical indicators (RSI, MACD, trend_score)
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import timm

from config import VIT_MODEL_NAME, NUM_CLASSES, CHECKPOINTS_DIR, BEST_CHECKPOINT

LEGACY_MODEL_NAME_BY_EMBED_DIM = {
    192: "vit_tiny_patch16_224",
    384: "vit_small_patch16_224",
    768: "vit_base_patch16_224",
}
FALLBACK_CHECKPOINT_NAMES = (
    BEST_CHECKPOINT,
    "best_model.pth",
    "bestmodel.pth",
)


class CandlestickViT(nn.Module):
    """
    Vision Transformer model for candlestick chart trend classification.

    The default head matches the current repo architecture. A legacy mode is
    also supported for older checkpoints trained with extra indicator inputs.
    """

    def __init__(
        self,
        model_name: str = VIT_MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        use_aux_features: bool = False,
        head_variant: str = "modern",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_aux_features = use_aux_features
        self.num_aux_features = 3 if use_aux_features else 0
        self.head_variant = head_variant

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        self.feature_dim = self.backbone.num_features
        classifier_input_dim = self.feature_dim + self.num_aux_features

        if head_variant == "legacy":
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(classifier_input_dim),
                nn.Dropout(dropout),
                nn.Linear(classifier_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(
        self,
        x: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224).
            aux_features: Optional tensor of shape (batch_size, 3).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        features = self.backbone(x)

        if self.use_aux_features:
            if aux_features is None:
                aux_features = torch.zeros(
                    (features.size(0), self.num_aux_features),
                    device=features.device,
                    dtype=features.dtype,
                )
            else:
                aux_features = aux_features.to(device=features.device, dtype=features.dtype)
            features = torch.cat([features, aux_features], dim=1)

        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations without the classifier."""
        return self.backbone(x)

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract attention maps from all transformer blocks."""
        attention_maps = []
        hooks = []

        def hook_fn(module, input, output):
            if hasattr(module, "attn_drop"):
                attention_maps.append(output)

        for block in self.backbone.blocks:
            hook = block.attn.attn_drop.register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            _ = self.backbone.forward_features(x)

        for hook in hooks:
            hook.remove()

        return attention_maps


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Normalize supported checkpoint objects to a raw state dict."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError("Unsupported checkpoint format.")


def _is_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """Return True for older Kaggle checkpoints using vit./fc. keys."""
    return any(key.startswith("vit.") for key in state_dict) and any(
        key.startswith("fc.") for key in state_dict
    )


def _build_model_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    pretrained_backbone: bool,
) -> tuple[CandlestickViT, dict[str, torch.Tensor]]:
    """Create a compatible model instance and translated state dict."""
    if _is_legacy_state_dict(state_dict):
        embed_dim = int(state_dict["vit.norm.weight"].shape[0])
        model_name = LEGACY_MODEL_NAME_BY_EMBED_DIM.get(embed_dim, VIT_MODEL_NAME)
        hidden_dim = int(state_dict["fc.0.weight"].shape[0])
        classifier_input_dim = int(state_dict["fc.0.weight"].shape[1])
        use_aux_features = classifier_input_dim > embed_dim

        model = CandlestickViT(
            model_name=model_name,
            pretrained=pretrained_backbone,
            hidden_dim=hidden_dim,
            use_aux_features=use_aux_features,
            head_variant="legacy",
        )

        translated_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("vit."):
                translated_state_dict[f"backbone.{key[4:]}"] = value
            elif key.startswith("fc."):
                translated_state_dict[f"classifier.{key[3:]}"] = value
            else:
                translated_state_dict[key] = value

        return model, translated_state_dict

    model = CandlestickViT(pretrained=pretrained_backbone)
    return model, state_dict


def load_model(
    checkpoint_path: Path | str | None = None,
    device: str = "cpu",
    pretrained_backbone: bool = True,
) -> CandlestickViT:
    """
    Load the CandlestickViT model, optionally from a checkpoint.

    Args:
        checkpoint_path: Path to a model checkpoint. If None, uses default.
        device: Device to load the model on ('cpu', 'cuda').
        pretrained_backbone: Whether to use pretrained backbone weights if no
            checkpoint is found.

    Returns:
        Loaded model in eval mode.
    """
    if checkpoint_path is None:
        checkpoint_candidates = [
            CHECKPOINTS_DIR / checkpoint_name
            for checkpoint_name in FALLBACK_CHECKPOINT_NAMES
        ]
        checkpoint_path = next(
            (path for path in checkpoint_candidates if path.exists()),
            checkpoint_candidates[0],
        )

    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        print(f"[OK] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = _extract_state_dict(checkpoint)
        model, compatible_state_dict = _build_model_from_state_dict(
            state_dict,
            pretrained_backbone=False,
        )
        model.load_state_dict(compatible_state_dict)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")
            print(f"   Val Acc: {checkpoint.get('val_acc', 'N/A')}")
    else:
        model = CandlestickViT(pretrained=pretrained_backbone)
        print(f"[WARN] No checkpoint found at: {checkpoint_path}")
        print("   Using pretrained backbone with untrained classification head.")
        print("   Predictions will be unreliable until model is fine-tuned.")

    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = CandlestickViT(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print(f"Feature dim: {model.feature_dim}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
