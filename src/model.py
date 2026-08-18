import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.labels import LABELS

# --image-sizeを指定せずに学習された重み(メタデータなしの旧形式)を読んだときに
# 前提とする入力サイズ。この値で学習された重みしか旧形式には存在しない。
DEFAULT_IMAGE_SIZE = 224


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


def save_checkpoint(path, model: nn.Module, image_size: int) -> None:
    """重みと一緒に、その重みが前提とする入力サイズ・ラベル一覧も保存する。

    EfficientNetは適応的プーリングを使うため、学習時と違う入力サイズを与えても
    エラーにならず「黙って精度が落ちる」。サイズを重みに同梱しておくことで、
    推論側が学習時と同じ前処理を自動で再現できるようにする。
    """
    torch.save(
        {"state_dict": model.state_dict(), "image_size": image_size, "labels": list(LABELS)},
        path,
    )


def load_checkpoint(path, model: nn.Module, map_location=None) -> dict:
    """save_checkpoint形式・旧形式(state_dictそのもの)のどちらも読めるローダー。

    重みをmodelに読み込んだうえで、メタデータ(image_size, labels)を返す。
    旧形式にはメタデータが無いため、DEFAULT_IMAGE_SIZEとLABELSを補って返す。
    """
    # 保存しているのはテンソル・int・文字列のみなので、torch 2.6以降の既定
    # (weights_only=True)でもそのまま読める。既定に任せてtorchのバージョン差を吸収する。
    obj = torch.load(path, map_location=map_location)

    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
        image_size = obj.get("image_size", DEFAULT_IMAGE_SIZE)
        labels = obj.get("labels", list(LABELS))
    else:  # 旧形式: state_dictがそのまま保存されている
        state_dict = obj
        image_size = DEFAULT_IMAGE_SIZE
        labels = list(LABELS)

    if labels != list(LABELS):
        raise ValueError(
            "重みのラベル構成が現在のsrc/labels.pyと一致しません。\n"
            f"  重み側: {labels}\n"
            f"  現在  : {list(LABELS)}\n"
            "ラベルを変更した場合は再学習が必要です。"
        )

    model.load_state_dict(state_dict)
    return {"image_size": image_size, "labels": labels}
