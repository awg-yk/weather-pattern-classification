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
    return {
        "site": site.name,
        "n_high": len(highs),
        "n_low": len(lows),
        "highs": highs,
        "lows": lows,
        "nearest": nearest,
        "n_high_all": len(detections.highs),
        "n_low_all": len(detections.lows),
    }


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
