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


class FeatureFusion(nn.Module):
    """CNNが画像から作った特徴ベクトルに、ERA5の数値特徴を連結して分類する。

    天気図の画像には「等圧線がどう曲がっているか」は写っているが、
    「オホーツク海の気圧が周囲より何hPa高いか」は数値としては読み取れない。
    ERA5の領域特徴(scripts/era5_features.py)を線形モデルに与えるだけで
    オホーツク海高気圧のAUCが0.879に達したことから、その情報は確かに存在し、
    画像からの抽出が難しいだけだと分かっている。そこで両方を分類器に渡す。

    数値特徴は学習データの平均・標準偏差で正規化してから使う。単位も桁も
    バラバラ(気圧はhPa、位置は度)のままだと、一部の特徴だけが極端に強く効くため。
    正規化に使う統計は重みと一緒に保存し、推論時も同じ値で揃える。
    """

    def __init__(self, net: nn.Module, num_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        inner = net.net if isinstance(net, CoordConv) else net
        in_features = inner.classifier[1].in_features
        # 画像側の分類ヘッドを外し、プーリング後の特徴ベクトルを取り出す
        inner.classifier = nn.Identity()

        self.net = net
        self.num_features = num_features
        # 正規化の統計。学習前に set_feature_stats() で入れ、重みと一緒に保存される。
        self.register_buffer("feature_mean", torch.zeros(num_features))
        self.register_buffer("feature_std", torch.ones(num_features))
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features + num_features, num_classes),
        )

    def set_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """学習データだけから求めた平均・標準偏差を設定する。

        検証・テストのデータを含めて計算すると、そこにしか無い情報が
        学習側に漏れるため、必ず学習用サブセットから求めること。
        """
        self.feature_mean.copy_(mean)
        # 値が一定の特徴で0除算にならないようにする
        self.feature_std.copy_(std.clamp(min=1e-6))

    def forward(self, x: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        if features is None:
            raise ValueError(
                "このモデルはERA5特徴量を使う構成です。forward(images, features)の形で"
                "呼んでください(--era5-features を指定して学習された重みです)。"
            )
        image_features = self.net(x)
        scaled = (features - self.feature_mean) / self.feature_std
        return self.head(torch.cat([image_features, scaled], dim=1))


class SmallCNN(nn.Module):
    """ERA5格子のための、浅くて単純な畳み込みネット。

    格子入力ではEfficientNet-B0が極端な過学習に陥る -- 学習側のAPが0.95に達する
    一方で検証側は横ばいのままで、エポック1の重みのほうがエポック24の重みより
    テストで良い。正しくベストエポックを選ぶと、自明な予測を下回る(macro F1 0.234
    に対し、全部を陽性と答えるだけで0.258)。

    当初これを容量の問題と考えたが、そうではなかった。幅を振ると性能は6万から
    385万パラメータまで単調に上がり続ける(0.386→0.497)。EfficientNet-B0と
    ほぼ同じ385万パラメータのこのネットが0.497を出す一方、EfficientNetは0.234。
    容量を揃えても結果が正反対になるので、効いているのは構造のほうである。

    EfficientNet側の何が悪いのかは、まだ切り分けていない。候補は、深さ(16ブロック
    対4段)、気圧場には転移しないImageNet重みからの出発、そして2チャンネル入力では
    意味をなさないSqueeze-and-Excitationブロック。

    幅はcnn_widthsで変えられる。既定の(128,256,512,512)は実測で最も良かった値で、
    fold間のばらつきも最小(標準偏差0.007)だった。

    構造はEfficientNetと同じ約束事に従う -- features / classifier という名前で
    公開し、features[0][0] を最初の畳み込みにする。CoordConv・FeatureFusion・
    save_checkpoint がその形を前提にしているため。
    """

    def __init__(self, num_classes: int, in_channels: int = 2, dropout: float = 0.3,
                 widths=(128, 256, 512, 512)):
        super().__init__()
        blocks = []
        previous = in_channels
        for width in widths:
            blocks.append(nn.Sequential(
                nn.Conv2d(previous, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            previous = width
        self.widths = tuple(widths)
        self.features = nn.Sequential(*blocks)
        # 大域平均プーリングで入力サイズに依存しなくする(EfficientNetと同じ)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(previous, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


def _adapt_first_conv(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    """最初の畳み込みを in_channels 入力に差し替える。事前学習重みは引き継ぐ。

    ImageNetの重みでチャンネル数に依存するのは、この最初の畳み込みだけである
    (3×32×3×3 = 864パラメータ)。EfficientNet-B0全体は約530万パラメータなので、
    形の合わない0.02%のために残り99.98%を捨てるのは、もったいない。
    2層目以降が学習しているエッジ・勾配・ブロブの検出は、RGB画像でなくても
    (気圧場のような滑らかな物理量であっても)初期値としてランダムよりは役に立つ。

    新しい層の重みは、RGB3チャンネル分の平均で全チャンネルを初期化する。
    衛星のマルチスペクトル画像で標準的に使われる方法で、チャンネル数を水増しして
    無い情報を捏造する代わりに、重みの側を入力の形に合わせる。さらに 3/in_channels
    を掛けることで、全チャンネルに同じ値を入れたときの出力が元と一致するようにし、
    後続の層が前提としている活性の大きさを崩さない。
    """
    first_conv = model.features[0][0]
    replacement = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            averaged = first_conv.weight.mean(dim=1, keepdim=True)
            replacement.weight.copy_(averaged.expand(-1, in_channels, -1, -1) * (3.0 / in_channels))
            if first_conv.bias is not None:
                replacement.bias.copy_(first_conv.bias)
        print(f"最初の畳み込みを{in_channels}チャンネル入力に作り直し、"
              "残りの層はImageNetの事前学習重みを引き継ぎます")
    model.features[0][0] = replacement


def build_model(
    num_classes: int = len(LABELS),
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
    coordconv: bool = False,
    num_features: int = 0,
    in_channels: int = 3,
    arch: str = "efficientnet_b0",
    cnn_widths=(128, 256, 512, 512),
) -> nn.Module:
    """EfficientNet-B0をベースにした転移学習モデル。データが少ない段階に適したサイズ。

    freeze_backbone=True にすると特徴抽出部を凍結し、分類ヘッドのみ学習する。
    データが数百枚程度と少ない場合、過学習を抑えるのに有効。

    coordconv=True にすると入力に座標チャンネルを足す(CoordConvを参照)。
    num_features>0 にするとERA5の数値特徴を併用する(FeatureFusionを参照)。

    arch="small_cnn" にすると、EfficientNetの代わりに小さな畳み込みネットを使う
    (SmallCNNを参照)。ERA5格子のようにデータ量が少なく事前学習も効かない入力向け。
    cnn_widthsで各段の幅を変えられる。容量が効くことが分かっている以上、既定の
    (32,64,128,128)が最適とは限らないため。

    in_channels!=3 は、天気図画像(RGB)ではなくERA5の格子(気圧・気温など)を
    直接入力するとき用(src/era5_grid.pyを参照)。このとき形が合わないのは
    最初の畳み込み1層だけなので、その層だけ作り直し、残りは事前学習重みを
    引き継ぐ(_adapt_first_conv を参照)。
    """
    if arch == "small_cnn":
        # 事前学習重みは存在しないので、pretrainedは無視する
        model = SmallCNN(num_classes, in_channels=in_channels, dropout=dropout,
                         widths=tuple(cnn_widths))
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        if coordconv:
            model = CoordConv(model)
        if num_features:
            model = FeatureFusion(model, num_features, num_classes, dropout=dropout)
        return model

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if in_channels != 3:
        _adapt_first_conv(model, in_channels, pretrained)

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
    if num_features:
        model = FeatureFusion(model, num_features, num_classes, dropout=dropout)
    return model


def backbone(model: nn.Module):
    """CoordConv・FeatureFusionで包まれていても中のEfficientNetを返す。Grad-CAM用。"""
    while isinstance(model, (CoordConv, FeatureFusion)):
        model = model.net
    return model


def _base_in_channels(model: nn.Module) -> int:
    """build_model()のin_channels引数に相当する値を、実際の層から逆算する。

    CoordConvは最初の畳み込みに+2チャンネル足すため、実際の層のin_channelsを
    そのまま記録すると、読み込み時にbuild_model(in_channels=..., coordconv=True)が
    もう一度+2してしまい、チャンネル数が二重に増えてしまう。ここで先に2を
    差し引いておくことで、save_checkpoint/load_checkpointの往復を正しく保つ。
    """
    channels = backbone(model).features[0][0].in_channels
    return channels - 2 if _has(model, CoordConv) else channels


def save_checkpoint(path, model: nn.Module, image_size: int, pos_weight=None) -> None:
    """重みと一緒に、その重みが前提とする入力サイズ・ラベル一覧も保存する。

    EfficientNetは適応的プーリングを使うため、学習時と違う入力サイズを与えても
    エラーにならず「黙って精度が落ちる」。サイズを重みに同梱しておくことで、
    推論側が学習時と同じ前処理を自動で再現できるようにする。

    pos_weightも同梱する。BCEWithLogitsLoss(pos_weight=w)で学習した出力は
    真の確率よりw倍ぶん高く出る(詳細はsrc/calibration.py)。この値が分かれば
    ロジットからlog wを引くだけで解析的に戻せるため、確信度の校正に必要になる。
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": image_size,
            "labels": list(LABELS),
            "pos_weight": None if pos_weight is None else [float(w) for w in pos_weight],
            "coordconv": _has(model, CoordConv),
            "num_features": model.num_features if isinstance(model, FeatureFusion) else 0,
            "in_channels": _base_in_channels(model),
            "arch": "small_cnn" if isinstance(backbone(model), SmallCNN) else "efficientnet_b0",
            # 幅が違えば別の形なので、記録しないと読み戻せない
            "cnn_widths": list(backbone(model).widths) if isinstance(backbone(model), SmallCNN) else [],
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
        num_features = obj.get("num_features", 0)
        in_channels = obj.get("in_channels", 3)
        # pos_weightを記録する前に作られた重みでは None になる。その場合は
        # 解析的な補正ができないので、校正は検証データへの当てはめだけで行う。
        pos_weight = obj.get("pos_weight")
    else:  # 旧形式: state_dictがそのまま保存されている
        state_dict = obj
        image_size = DEFAULT_IMAGE_SIZE
        labels = list(LABELS)
        coordconv = False
        num_features = 0
        in_channels = 3
        pos_weight = None

    if labels != list(LABELS):
        raise ValueError(
            "重みのラベル構成が現在のsrc/labels.pyと一致しません。\n"
            f"  重み側: {labels}\n"
            f"  現在  : {list(LABELS)}\n"
            "ラベルを変更した場合は再学習が必要です。"
        )

    model_features = model.num_features if isinstance(model, FeatureFusion) else 0
    model_in_channels = _base_in_channels(model)
    if (
        coordconv != _has(model, CoordConv)
        or num_features != model_features
        or in_channels != model_in_channels
    ):
        raise ValueError(
            "重みの構成と渡されたモデルが一致しません。\n"
            f"  重み側  : coordconv={coordconv}, num_features={num_features}, in_channels={in_channels}\n"
            f"  モデル側: coordconv={_has(model, CoordConv)}, num_features={model_features}, "
            f"in_channels={model_in_channels}\n"
            "build_modelではなくload_modelを使うと、重みに合わせて自動で組み立てます。"
        )

    model.load_state_dict(state_dict)
    return {
        "image_size": image_size,
        "labels": labels,
        "coordconv": coordconv,
        "num_features": num_features,
        "in_channels": in_channels,
        "pos_weight": pos_weight,
    }


def _has(model: nn.Module, kind) -> bool:
    """CoordConvがFeatureFusionの内側にあっても見つけられるようにする。"""
    while isinstance(model, (CoordConv, FeatureFusion)):
        if isinstance(model, kind):
            return True
        model = model.net
    return False


def load_model(path, map_location=None):
    """重みのメタデータに合わせてモデルを組み立ててから読み込む。

    CoordConvの有無は重みごとに違うため、推論側が構成を知らなくても
    正しいモデルを復元できるようにこの関数を通す。(model, メタデータ)を返す。
    """
    obj = torch.load(path, map_location=map_location)
    meta_obj = obj if isinstance(obj, dict) else {}
    model = build_model(
        num_classes=len(LABELS),
        pretrained=False,
        coordconv=meta_obj.get("coordconv", False),
        num_features=meta_obj.get("num_features", 0),
        in_channels=meta_obj.get("in_channels", 3),
        arch=meta_obj.get("arch", "efficientnet_b0"),
        cnn_widths=meta_obj.get("cnn_widths") or (128, 256, 512, 512),
    )
    meta = load_checkpoint(path, model, map_location=map_location)
    return model, meta
