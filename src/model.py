import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.labels import LABELS


def build_model(
    num_classes: int = len(LABELS),
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    """EfficientNet-B0をベースにした転移学習モデル。データが少ない段階に適したサイズ。

    freeze_backbone=True にすると特徴抽出部を凍結し、分類ヘッドのみ学習する。
    データが数百枚程度と少ない場合、過学習を抑えるのに有効。
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
