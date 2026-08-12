import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.labels import LABELS


def build_model(num_classes: int = len(LABELS), pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0をベースにした転移学習モデル。データが少ない段階に適したサイズ。"""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
