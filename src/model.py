import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.labels import LABELS

# --image-sizeを指定せずに学習された重み(メタデータなしの旧形式)を読んだときに
# 前提とする入力サイズ。この値で学習された重みしか旧形式には存在しない。
DEFAULT_IMAGE_SIZE = 224


class CoordConv(nn.Module):
    """入力画像に「そのピクセルが図のどこにあるか」を表す2チャンネルを足すラッパー。

    畳み込みは平行移動に対して同じ反応をするため、「日本海にある低気圧」と
    「オホーツク海にある高気圧」のように"位置そのものが定義に含まれる"気圧配置を
    区別しづらい。Grad-CAMでも、オホーツク海高気圧の判定時にモデルが高気圧本体では
    なく下流の等圧線を見ていることが確認できた。

    そこで x座標・y座標を-1〜1に正規化した2枚のマップを画像に連結し、
    畳み込みが絶対位置を直接参照できるようにする(Liu et al., 2018 "CoordConv")。
    天気図は常に同じ図法・同じ描画範囲なので、画像上の座標はそのまま緯度経度に対応する。

    追加した2チャンネル分の重みは0で初期化するので、学習開始時点の出力は
    元の学習済みモデルと完全に一致する。位置情報を使うかどうかは学習が決める。
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        first_conv = net.features[0][0]
        expanded = nn.Conv2d(
            first_conv.in_channels + 2,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, : first_conv.in_channels] = first_conv.weight
            if first_conv.bias is not None:
                expanded.bias.copy_(first_conv.bias)
        net.features[0][0] = expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        grid_y = ys.view(1, 1, h, 1).expand(n, 1, h, w)
        grid_x = xs.view(1, 1, 1, w).expand(n, 1, h, w)
        return self.net(torch.cat([x, grid_y, grid_x], dim=1))


def build_model(
    num_classes: int = len(LABELS),
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
    coordconv: bool = False,
) -> nn.Module:
    """EfficientNet-B0をベースにした転移学習モデル。データが少ない段階に適したサイズ。

    freeze_backbone=True にすると特徴抽出部を凍結し、分類ヘッドのみ学習する。
    データが数百枚程度と少ない場合、過学習を抑えるのに有効。

    coordconv=True にすると入力に座標チャンネルを足す(CoordConvを参照)。
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
    if coordconv:
        model = CoordConv(model)
    return model


def backbone(model: nn.Module):
    """CoordConvで包まれていても中のEfficientNetを返す。Grad-CAM用。"""
    return model.net if isinstance(model, CoordConv) else model


def save_checkpoint(path, model: nn.Module, image_size: int) -> None:
    """重みと一緒に、その重みが前提とする入力サイズ・ラベル一覧も保存する。

    EfficientNetは適応的プーリングを使うため、学習時と違う入力サイズを与えても
    エラーにならず「黙って精度が落ちる」。サイズを重みに同梱しておくことで、
    推論側が学習時と同じ前処理を自動で再現できるようにする。
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": image_size,
            "labels": list(LABELS),
            "coordconv": isinstance(model, CoordConv),
        },
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
        coordconv = obj.get("coordconv", False)
    else:  # 旧形式: state_dictがそのまま保存されている
        state_dict = obj
        image_size = DEFAULT_IMAGE_SIZE
        labels = list(LABELS)
        coordconv = False

    if labels != list(LABELS):
        raise ValueError(
            "重みのラベル構成が現在のsrc/labels.pyと一致しません。\n"
            f"  重み側: {labels}\n"
            f"  現在  : {list(LABELS)}\n"
            "ラベルを変更した場合は再学習が必要です。"
        )

    if coordconv != isinstance(model, CoordConv):
        raise ValueError(
            "重みの構成と渡されたモデルが一致しません"
            f"(重み側 coordconv={coordconv} / モデル側 coordconv={not coordconv})。\n"
            "build_modelではなくload_modelを使うと、重みに合わせて自動で組み立てます。"
        )

    model.load_state_dict(state_dict)
    return {"image_size": image_size, "labels": labels, "coordconv": coordconv}


def load_model(path, map_location=None):
    """重みのメタデータに合わせてモデルを組み立ててから読み込む。

    CoordConvの有無は重みごとに違うため、推論側が構成を知らなくても
    正しいモデルを復元できるようにこの関数を通す。(model, メタデータ)を返す。
    """
    obj = torch.load(path, map_location=map_location)
    coordconv = obj.get("coordconv", False) if isinstance(obj, dict) else False
    model = build_model(num_classes=len(LABELS), pretrained=False, coordconv=coordconv)
    meta = load_checkpoint(path, model, map_location=map_location)
    return model, meta
