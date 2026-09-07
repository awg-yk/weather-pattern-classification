"""利用者が指定した地点と、その周辺だけを見るための部品。

背景
----
モデルは日本全体の天気図を見て気圧配置を判断する。そのため、小松のおろし風の
ように**特定の地点で起きる現象**を調べると、その地点に影響しない遠方の気圧配置
まで拾ってしまう(167日の検証では17件がこれに当たった)。

ここでは天気図上に「調べたい地点とその周辺」を1つの円として定義し、検出した
高気圧・低気圧をその円の中だけに絞り込めるようにする。

`src/regions.py` の `Region` との違い
------------------------------------
`Region` は**ラベルごと**の「見るべき領域」で、名前が `src/labels.py` の
LABELS に無いと例外になる。こちらは**利用者が決める地点**なので、名前は自由で、
形は矩形ではなく円(中心と半径)にしてある。地点からの距離で切るほうが、
「小松の周辺」という言い方にそのまま対応するため。

座標系
------
`src/regions.py` と同じ相対座標。画像の左上を(0,0)、右下を(1,1)とし、
xが列(左→右)、yが行(上→下)。`ChartDetections.highs` / `.lows` が持つ
座標もこれと同じなので、そのまま比較できる。

**緯度経度からは変換しない。**天気図は正距円筒図法ではなく、前処理の
autocrop_to_content() が白縁を落とすため画素と緯度経度の対応表も無い。
線形変換で置くと嘘の精度が付く(`docs/2026-08-25-attention-regions.md` に、
経緯線の目盛りから作った式で東京が本州北部に載った記録がある)。
**地点の座標は天気図に重ねて目で決める。**scripts/site_preview.py を使うこと。
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_SITES_PATH = Path(__file__).resolve().parent.parent / "data" / "sites.csv"

SITE_COLUMNS = ("name", "x", "y", "radius", "note")

# 半径の既定値(相対座標)。天気図の幅はおよそ経度40度ぶんなので、
# 0.08 は経度約3.2度、緯度36度付近で約290kmにあたる。総観規模の系が
# その地点の天気に効く範囲としておおよその目安になる。
# **目安であって、気象学的に導いた値ではない。**用途に応じて --radius で変える。
DEFAULT_RADIUS = 0.08


@dataclass(frozen=True)
class Site:
    """調べたい地点と、その周辺とみなす範囲。相対座標(0〜1)の円。"""

    name: str
    x: float
    y: float
    radius: float = DEFAULT_RADIUS
    note: str = ""

    def __post_init__(self):
        for field_name, value in (("x", self.x), ("y", self.y)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{self.name}: {field_name}={value} が相対座標(0〜1)の範囲外です。")
        if not 0.0 < float(self.radius) <= 1.0:
            raise ValueError(
                f"{self.name}: radius={self.radius} は0より大きく1以下にしてください。")

    def distance(self, x: float, y: float) -> float:
        """地点から点までの距離(相対座標)。"""
        return math.hypot(float(x) - self.x, float(y) - self.y)

    def contains(self, x: float, y: float) -> bool:
        """その点が地点の周辺(円の中)にあるか。"""
        return self.distance(x, y) <= self.radius

    @property
    def area(self) -> float:
        """画像全体に対する円の面積比。注目が一様なときの mass の期待値。

        相対座標のうえでの円なので、面積は pi*r^2。画素の上では縦横比のぶん
        楕円に見えるが、判定も面積もすべて相対座標で揃えてある。
        """
        return math.pi * self.radius ** 2

    def mask(self, height: int, width: int, supersample: int = 4) -> np.ndarray:
        """円の内側を1、外側を0にした (height, width) の重み。

        境界の画素は、その画素のうち円に入っている割合を持つ。
        **0/1で切ると、格子の細かさを変えただけで数値が動いてしまう**
        (src/regions.py の Region.mask と同じ理由)。境界だけを細かく
        数え直すのは面倒なので、全体を supersample 倍で数えて平均する。
        """
        fine_h, fine_w = height * supersample, width * supersample
        ys = (np.arange(fine_h) + 0.5) / fine_h
        xs = (np.arange(fine_w) + 0.5) / fine_w
        inside = ((xs[None, :] - self.x) ** 2
                  + (ys[:, None] - self.y) ** 2) <= self.radius ** 2
        return (inside.astype(np.float32)
                .reshape(height, supersample, width, supersample)
                .mean(axis=(1, 3)))

    def distance_map(self, height: int, width: int) -> np.ndarray:
        """各画素の中心から地点までの距離(相対座標)。"""
        ys = (np.arange(height) + 0.5) / height
        xs = (np.arange(width) + 0.5) / width
        return np.hypot(xs[None, :] - self.x, ys[:, None] - self.y)

    def weight_map(self, height: int, width: int, scale: float = None) -> np.ndarray:
        """地点に近いほど大きい重み(0〜1)。既定の scale は半径。

        円のマスク(mask)と違い、境界で切らずに滑らかに落ちる。
        """
        scale = self.radius if scale is None else scale
        return proximity_weight(self.distance_map(height, width), scale)

    def pixel_circle(self, width: int, height: int) -> tuple:
        """(left, top, right, bottom) を画素で返す。描画用。

        画像は縦横比が1ではないので、相対座標の円は画素の上では楕円になる。
        判定は相対座標で行うため、見た目を判定に合わせてこう描く。
        """
        return (
            int(round((self.x - self.radius) * width)),
            int(round((self.y - self.radius) * height)),
            int(round((self.x + self.radius) * width)),
            int(round((self.y + self.radius) * height)),
        )


def load_sites(path=None) -> dict:
    """data/sites.csv を読んで {地点名: Site} を返す。"""
    path = Path(path) if path else DEFAULT_SITES_PATH
    if not path.exists():
        raise SystemExit(
            f"地点の定義がありません: {path}\n"
            "  name,x,y,radius,note の5列のCSVを置いてください"
            "(x,y,radiusは相対座標)")
    table = pd.read_csv(path)
    missing = [c for c in SITE_COLUMNS if c not in table.columns]
    if missing:
        raise SystemExit(f"{path} に列が足りません: {missing}")

    sites = {}
    for row in table.itertuples(index=False):
        site = Site(name=str(row.name), x=float(row.x), y=float(row.y),
                    radius=float(row.radius),
                    note="" if pd.isna(row.note) else str(row.note))
        sites[site.name] = site
    return sites


def get_site(name: str, path=None) -> Site:
    """名前で1つ引く。無ければ、あるものを並べて教える。"""
    sites = load_sites(path)
    if name not in sites:
        raise SystemExit(
            f"地点 {name!r} が見つかりません。定義されているのは: "
            + "、".join(sites) + "\n"
            "  新しい地点は data/sites.csv に1行足してください"
            "(位置は scripts/site_preview.py で確かめること)")
    return sites[name]


def nearby(points: list, site: Site) -> list:
    """検出した点のうち、地点の周辺にあるものだけを返す。

    points は `ChartDetections.highs` / `.lows` の形((x, y) の並び)。
    地点に近い順に並べ替えて返すので、先頭が最も近い系になる。
    """
    inside = [p for p in points if site.contains(p[0], p[1])]
    return sorted(inside, key=lambda p: site.distance(p[0], p[1]))


def summarize(detections, site: Site) -> dict:
    """1枚ぶんの検出結果を、地点の周辺に絞って数える。

    **中心が枠外の系(edge_highs / edge_lows)は数えない。**それらは文字の
    位置しか分かっておらず中心ではないので、距離で切ると嘘になる
    (`src/chartfeatures.py` の「中心が枠外の系について」を参照)。
    """
    highs = nearby(detections.highs, site)
    lows = nearby(detections.lows, site)
    nearest = None
    for kind, points in (("H", highs), ("L", lows)):
        for point in points:
            distance = site.distance(point[0], point[1])
            if nearest is None or distance < nearest["distance"]:
                nearest = {"kind": kind, "x": point[0], "y": point[1],
                           "distance": distance}
    # 円の内外だけでなく、**距離で滑らかに重み付けした点数**も出す。
    # 円は境界のすぐ外にある系を捨ててしまい、0の日ばかりになる
    scale = site.radius
    nearness = sum(
        float(proximity_weight(site.distance(p[0], p[1]), scale))
        for p in list(detections.highs) + list(detections.lows)
    )
    return {
        "site": site.name,
        "n_high": len(highs),
        "n_low": len(lows),
        "nearness": nearness,
        "highs": highs,
        "lows": lows,
        "nearest": nearest,
        "n_high_all": len(detections.highs),
        "n_low_all": len(detections.lows),
    }


def proximity_weight(distance, scale: float):
    """地点からの距離を 0〜1 の重みに直す(ガウス)。

        w = exp(-(distance / scale) ** 2)

    `scale` のところで重みが約0.37、その倍の距離で約0.018になる。

    **円で「中か外か」を切るのをやめるための関数。**円は境界のすぐ外にある
    ものを全部捨てる。小松のおろし風167日では、半径0.12の円に高低気圧が
    1つも入らない日が85.6%あり、そのうち大半は「円が天気図の4.5%しか
    ないから」で説明がついてしまった。距離で滑らかに落とせば崖が無くなる。

    **面積の目盛りは円と揃っている。**この重みを平面全体で積分すると
    pi * scale^2 になり、半径 scale の円の面積と一致する。だから
    scale=半径 とすれば、下の lift は円のときと同じ意味で読める。
    """
    return np.exp(-(np.asarray(distance, dtype="float64") / scale) ** 2)


def attention_proximity(cam, site: Site, scale: float = None,
                        size: int = None) -> float:
    """Grad-CAMの熱を、地点からの近さで重み付けして合計した割合。

    `attention_mass` の円を、距離で滑らかに落ちる重みに置き換えたもの。
    熱が全く無いCAMでは割合が決まらないので0.0を返す。
    """
    from src.regions import CAM_GRID, resize_cam

    size = CAM_GRID if size is None else size
    scale = site.radius if scale is None else scale
    grid = resize_cam(cam, size)
    total = float(grid.sum())
    if total <= 0:
        return 0.0
    return float((grid * site.weight_map(size, size, scale)).sum() / total)


def proximity_lift(cam, site: Site, scale: float = None, size: int = None) -> float:
    """近さで重み付けした割合 / 一様に見たときの期待値。

    1なら画像全体を一様に見ているのと同じ、1より大きいほど地点の近くに
    集中している。**円のときの lift と同じ意味**で読める(上の注記を参照)。
    """
    from src.regions import CAM_GRID

    size = CAM_GRID if size is None else size
    scale = site.radius if scale is None else scale
    uniform = float(site.weight_map(size, size, scale).mean())
    if uniform <= 0:
        return 0.0
    return attention_proximity(cam, site, scale, size) / uniform


def attention_mass(cam, site: Site, size: int = None) -> float:
    """Grad-CAMの熱のうち、地点の円の中に入っている割合(0〜1)。

    `src/regions.py` の同名の関数と同じ measure を、矩形ではなく円で行う。
    熱が全く無いCAMでは割合が決まらないので0.0を返す。
    """
    from src.regions import CAM_GRID, resize_cam

    size = CAM_GRID if size is None else size
    grid = resize_cam(cam, size)
    total = float(grid.sum())
    if total <= 0:
        return 0.0
    return float((grid * site.mask(size, size)).sum() / total)


def attention_lift(cam, site: Site, size: int = None) -> float:
    """mass / area。1なら一様に見ているのと同じ、1より大きいほど円に集中している。

    **順位を付け替えるだけなら mass と lift は同じ結果になる。**同じ天気図・
    同じ円ならラベルによらず area は一定なので、割っても順位は変わらない。
    絶対値を「集中しているか」として読みたいときに lift を使う。
    """
    return attention_mass(cam, site, size) / site.area


# 検出した系の印の色。annotate_charts.py と揃えている
HIGH_COLOR = (0, 160, 0)
LOW_COLOR = (200, 0, 0)
FAR_COLOR = (170, 170, 170)


def draw_detections(image: Image.Image, detections, site: Site, *,
                    arm: int = 14) -> Image.Image:
    """検出した高気圧・低気圧に印を打つ。円の内側は色付き、外側は灰色。

    **描き方はここ1か所に置く。**同じ絵をノートブックとコマンドの両方で描くので、
    書き写すと片方だけ直して食い違う(この計画では実際にその失敗をしている)。
    """
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for points, color in ((detections.highs, HIGH_COLOR), (detections.lows, LOW_COLOR)):
        for point in points:
            inside = site.contains(point[0], point[1])
            x = point[0] * width
            y = point[1] * height
            line_width = 4 if inside else 2
            shown = color if inside else FAR_COLOR
            draw.line([x - arm, y - arm, x + arm, y + arm], fill=shown, width=line_width)
            draw.line([x - arm, y + arm, x + arm, y - arm], fill=shown, width=line_width)
    return canvas


def draw_grid(image: Image.Image, step: float = 0.05) -> Image.Image:
    """相対座標の目盛りを重ねる。地点の位置を読み取って書き写すために使う。

    **緯度経度からは変換しない。**この天気図は正距円筒図法ではないので、
    経緯線の目盛りから作った線形の式は図の中央でずれる(実際に東京が本州北部に
    載った)。地点はこの目盛りを見て地図の上で直接読む。
    """
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    faint, strong = (150, 150, 150), (90, 90, 90)

    value = 0.0
    while value <= 1.0001:
        # 0.1刻みは濃く、その間は薄く。数えやすくするため
        major = abs(round(value / 0.1) * 0.1 - value) < 1e-9
        color = strong if major else faint
        x = int(round(value * width))
        y = int(round(value * height))
        draw.line([x, 0, x, height], fill=color, width=2 if major else 1)
        draw.line([0, y, width, y], fill=color, width=2 if major else 1)
        if major:
            draw.text((x + 4, 6), f"x={value:.1f}", fill=strong)
            draw.text((6, y + 4), f"y={value:.1f}", fill=strong)
        value += step
    return canvas


def draw_site(image: Image.Image, site: Site, color=(0, 140, 255), width: int = 4,
              text: str = None, font=None) -> Image.Image:
    """天気図に地点の円を描いて返す(元の画像は変更しない)。"""
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = site.pixel_circle(*canvas.size)
    draw.ellipse([left, top, right, bottom], outline=color, width=width)

    # 中心も打つ。円だけだと、地点そのものがどこかが読み取りにくい
    cx = int(round(site.x * canvas.size[0]))
    cy = int(round(site.y * canvas.size[1]))
    arm = max(6, width * 3)
    draw.line([cx - arm, cy, cx + arm, cy], fill=color, width=width)
    draw.line([cx, cy - arm, cx, cy + arm], fill=color, width=width)

    if text:
        draw.text((cx + arm + 4, cy + arm + 4), text, fill=color, font=font)
    return canvas
