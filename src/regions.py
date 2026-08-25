"""ラベルごとの「見るべき領域」を天気図上の矩形として持ち、Grad-CAMと突き合わせる。

背景
----
教師データ(data/labels_v2.csv)は画像1枚に対してラベル名を並べるだけで、
位置の情報を持っていない。一方でモデルの答えはGrad-CAMで「どこを見たか」まで
見える。この非対称のせいで、「オホーツク海高気圧と答えたのに本州中部を見ている」
といった誤りを、人が1枚ずつ目で見て指摘するしかなかった。

ここでは逆側に最低限の位置情報を与える。ラベルごとに1つの矩形
(data/regions.csv)を定義しておけば、Grad-CAMの熱がその中に何割入っているかを
数値にできる。1枚ずつの印象論が、ラベル別の1つの数字になる。

座標系
------
画像の左上を(0,0)、右下を(1,1)とする相対座標。scripts/preprocess_jma.py の
--stamp-box と同じ取り方で、xが列(左→右)、yが行(上→下)。

緯度経度ではなく相対座標を使う理由: 前処理の autocrop_to_content() が白縁を
落とすため画素と緯度経度の対応表が無く、また天気図は正距円筒図法ではないので
線形変換で緯度経度に直すと嘘の精度が付く。全画像が同じ基準でトリミングされて
いるので、相対座標なら画像間で揃う。矩形が実際の海域と合っているかは
scripts/regions_preview.py で天気図に重ねて目で確認する。

指標
----
mass   : Grad-CAMの総和のうち矩形の中にある割合。0〜1。
area   : 矩形が画像に占める面積の割合。注目が一様なときの mass の期待値。
lift   : mass / area。1なら「たまたま広いから入っているだけ」、
         1より大きいほどその領域に集中している。西高東低のように矩形が広い
         ラベルは mass だけ見ると高く出るので、必ず lift と併せて読む。
peak   : Grad-CAMが最大の画素が矩形の中にあるか(pointing game)。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.labels import LABELS

DEFAULT_REGIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "regions.csv"

# Grad-CAMを突き合わせ前に拡大する一辺の画素数。
# EfficientNet-B0の最終畳み込みは224px入力で7x7しかなく、そのまま数えると
# 矩形の境界が1/7刻みに丸まる。拡大しても情報は増えないが、境界をまたぐ画素の
# 熱を面積で按分できるので、矩形の指定がそのまま効くようになる。
CAM_GRID = 224

REGION_COLUMNS = ("label", "x0", "y0", "x1", "y1", "note")


@dataclass(frozen=True)
class Region:
    """1ラベルぶんの「見るべき領域」。相対座標(0〜1)の矩形。"""

    label: str
    x0: float
    y0: float
    x1: float
    y1: float
    note: str = ""

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(
                f"未知のラベルです: {self.label!r}。src/labels.py の LABELS にある名前を使ってください。"
            )
        for name, value in (("x0", self.x0), ("y0", self.y0), ("x1", self.x1), ("y1", self.y1)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{self.label}: {name}={value} が相対座標(0〜1)の範囲外です。")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                f"{self.label}: 矩形の幅か高さが0以下です "
                f"(x0={self.x0}, x1={self.x1}, y0={self.y0}, y1={self.y1})。"
            )

    @property
    def area(self) -> float:
        """画像全体に対する矩形の面積比。注目が一様なときの mass の期待値。"""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def pixel_box(self, width: int, height: int) -> tuple:
        """(left, top, right, bottom) を画素で返す。描画用。"""
        return (
            int(round(self.x0 * width)),
            int(round(self.y0 * height)),
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
        )

    def mask(self, height: int, width: int) -> np.ndarray:
        """矩形の内側を1、外側を0にした (height, width) の重み。

        境界をまたぐ画素は、その画素のうち矩形に入っている面積の割合を持つ。
        0/1で切ると、CAM_GRIDを変えただけで数値が動いてしまうため。
        """
        xs = _overlap_1d(width, self.x0, self.x1)
        ys = _overlap_1d(height, self.y0, self.y1)
        return ys[:, None] * xs[None, :]

    def contains(self, x: float, y: float) -> bool:
        """相対座標の点が矩形の中にあるか。"""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


def _overlap_1d(n: int, lo: float, hi: float) -> np.ndarray:
    """n分割した区間 [0,1] の各画素が [lo, hi] と重なる割合。"""
    edges = np.linspace(0.0, 1.0, n + 1)
    left = np.maximum(edges[:-1], lo)
    right = np.minimum(edges[1:], hi)
    return np.clip(right - left, 0.0, None) * n


def load_regions(path=None) -> dict:
    """data/regions.csv を読んで {ラベル名: Region} を返す。

    全ラベルが揃っていなくてもよい(領域が決まっていないラベルは測らないだけ)。
    不正な矩形や未知のラベル名はその場で例外にする -- 座標を打ち間違えたまま
    測り続けると、出た数値が何を意味するのか後から分からなくなるため。
    """
    path = Path(path) if path is not None else DEFAULT_REGIONS_PATH
    if not path.exists():
        raise FileNotFoundError(f"領域の定義ファイルがありません: {path}")

    frame = pd.read_csv(path)
    missing_columns = [c for c in REGION_COLUMNS[:5] if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"{path} に列がありません: {missing_columns}")

    regions = {}
    for row in frame.itertuples():
        region = Region(
            label=str(row.label),
            x0=float(row.x0),
            y0=float(row.y0),
            x1=float(row.x1),
            y1=float(row.y1),
            note=str(getattr(row, "note", "") or ""),
        )
        if region.label in regions:
            raise ValueError(f"{path}: {region.label} が2回定義されています。")
        regions[region.label] = region
    return regions


def resize_cam(cam: np.ndarray, size: int = CAM_GRID) -> np.ndarray:
    """Grad-CAMを (size, size) に双線形で拡大する。CAM_GRID の説明を参照。"""
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, None)
    if cam.shape == (size, size):
        return cam
    peak = float(cam.max())
    if peak <= 0:
        return np.zeros((size, size), dtype=np.float32)
    scaled = Image.fromarray((cam / peak * 255.0).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    return np.asarray(scaled, dtype=np.float32) / 255.0 * peak


def attention_mass(cam: np.ndarray, region: Region, size: int = CAM_GRID) -> float:
    """Grad-CAMの総和のうち矩形の中にある割合(0〜1)。

    熱が全く無い(全て0の)CAMでは割合が決まらないので0.0を返す。
    """
    grid = resize_cam(cam, size)
    total = float(grid.sum())
    if total <= 0:
        return 0.0
    return float((grid * region.mask(size, size)).sum() / total)


def attention_lift(cam: np.ndarray, region: Region, size: int = CAM_GRID) -> float:
    """mass / area。1なら一様注目と同じ、1より大きいほど矩形に集中している。"""
    return attention_mass(cam, region, size) / region.area


def peak_position(cam: np.ndarray) -> tuple:
    """Grad-CAMが最大の画素の位置を相対座標 (x, y) で返す。"""
    grid = np.clip(np.asarray(cam, dtype=np.float32), 0.0, None)
    row, col = np.unravel_index(int(np.argmax(grid)), grid.shape)
    height, width = grid.shape
    return ((col + 0.5) / width, (row + 0.5) / height)


def peak_in_region(cam: np.ndarray, region: Region) -> bool:
    """最も強く見ている点が矩形の中にあるか(pointing game)。"""
    x, y = peak_position(cam)
    return region.contains(x, y)


def draw_region(image: Image.Image, region: Region, color=(220, 30, 30), width: int = 3,
                text: str = None, font=None) -> Image.Image:
    """天気図に矩形を描いて返す(元の画像は変更しない)。"""
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = region.pixel_box(*canvas.size)
    draw.rectangle([left, top, right, bottom], outline=color, width=width)
    if text:
        draw.text((left + width + 2, top + width + 2), text, fill=color, font=font)
    return canvas
