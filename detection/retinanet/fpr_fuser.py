#!/usr/bin/env python3
"""
Learned fusers for Stage-2 scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

BASIC_FEATURES = ["det_score", "fpr_prob", "interaction"]
EXTENDED_FEATURES = [
    "det_score",
    "fpr_prob",
    "interaction",
    "log_volume",
    "elongation",
    "patch_mean",
    "patch_std",
    "patch_p90",
]


class LinearFPRFuser(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(1)


class MLPFPRFuser(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (32, 16), dropout: float = 0.1):
        super().__init__()
        dims = [input_dim, *[int(h) for h in hidden_dims if int(h) > 0], 1]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


def _safe_feature_tensor(extra_features: Optional[Dict[str, torch.Tensor]], key: str, ref: torch.Tensor) -> torch.Tensor:
    if extra_features is None or key not in extra_features:
        return torch.zeros_like(ref)
    return extra_features[key].float().to(ref.device)


def build_features(
    det_scores: torch.Tensor,
    fpr_probs: torch.Tensor,
    extra_features: Optional[Dict[str, torch.Tensor]] = None,
    feature_names: Optional[Iterable[str]] = None,
) -> torch.Tensor:
    det_scores = det_scores.float()
    fpr_probs = fpr_probs.float()
    feature_names = list(feature_names or BASIC_FEATURES)

    base = {
        "det_score": det_scores,
        "fpr_prob": fpr_probs,
        "interaction": det_scores * fpr_probs,
        "log_volume": _safe_feature_tensor(extra_features, "log_volume", det_scores),
        "elongation": _safe_feature_tensor(extra_features, "elongation", det_scores),
        "patch_mean": _safe_feature_tensor(extra_features, "patch_mean", det_scores),
        "patch_std": _safe_feature_tensor(extra_features, "patch_std", det_scores),
        "patch_p90": _safe_feature_tensor(extra_features, "patch_p90", det_scores),
    }
    missing = [name for name in feature_names if name not in base]
    if missing:
        raise ValueError(f"Unsupported fuser features: {missing}")
    return torch.stack([base[name] for name in feature_names], dim=1)


def build_fpr_fuser(
    model_arch: str,
    input_dim: int,
    hidden_dims: Iterable[int] = (32, 16),
    dropout: float = 0.1,
) -> nn.Module:
    arch = str(model_arch).strip().lower()
    if arch == "linear":
        return LinearFPRFuser(input_dim=input_dim)
    if arch == "mlp":
        return MLPFPRFuser(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    raise ValueError(f"Unsupported fuser architecture: {model_arch}")


def load_fpr_fuser(checkpoint_path: str, device: str = "cuda") -> Dict:
    """
    Load learned fuser checkpoint.
    Returns dict: {"model": nn.Module, "meta": dict}
    """
    ckpt_path = Path(checkpoint_path)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    feature_names = state.get("feature_names", BASIC_FEATURES)
    model_arch = state.get("model_arch", "linear")
    hidden_dims = state.get("hidden_dims", [32, 16])
    dropout = float(state.get("dropout", 0.1))

    model = build_fpr_fuser(
        model_arch=model_arch,
        input_dim=len(feature_names),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    meta = {
        "best_threshold": float(state.get("best_threshold", 0.5)),
        "feature_names": feature_names,
        "val_metrics": state.get("val_metrics", {}),
        "model_arch": model_arch,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
    }
    return {"model": model, "meta": meta}


def predict_fused_prob(
    fuser: Dict,
    det_scores: torch.Tensor,
    fpr_probs: torch.Tensor,
    device: torch.device,
    extra_features: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    model = fuser["model"]
    feature_names = fuser.get("meta", {}).get("feature_names", BASIC_FEATURES)
    feats = build_features(
        det_scores=det_scores,
        fpr_probs=fpr_probs,
        extra_features=extra_features,
        feature_names=feature_names,
    ).to(device)
    with torch.no_grad():
        logits = model(feats)
        probs = torch.sigmoid(logits)
    return probs
