"""天気図から取り出した記号と前線を、Phase 4 に渡す特徴量に変換する。

背景
----
`docs/2026-08-26-detection-plan.md` の Phase 3。物体検出の結果から
「高気圧数 / 低気圧数 / 高低気圧の位置 / 高低気圧間距離 / 前線本数 /
前線長 / 地域ごとの高低気圧存在有無」を作る。

位置は `src/regions.py` と同じ相対座標(0〜1)で受け取る。
`docs/2026-08-26-detection-prescreen.md` で、天気図が画素単位で揃っている
(画像間のずれ0.01画素)ことを確かめてあるので、相対座標は画像をまたいで
比較できる。

欠測をNaNのままにする理由
-------------------------
高気圧が1つも無い日には「高気圧の位置」が存在しない。0で埋めると図の
左上に高気圧があることになり、嘘の位置を教えることになる。NaNのまま
渡し、木のほうで「値が無い」枝として扱わせる
(`sklearn.ensemble.HistGradientBoostingClassifier` はNaNを直接扱える)。

中心が枠外の系について
----------------------
中心が図郭の外にある高気圧・低気圧には×が描かれず、文字だけになる
(`docs/2026-08-26-detection-prescreen.md`)。文字の位置は中心ではないので、
矩形の内外判定に使うと嘘になる。`edge_high` / `edge_low` は「図の縁に
記号がある」ことだけを伝える特徴量で、位置そのものは主張しない。
"""

import math
from dataclasses import dataclass, field

import numpy as np

from src.labels import LABELS

# 記号が図の縁からこの割合の内側までにあれば「縁にある」とみなす。
# 中心が枠外の系は、文字だけが縁の内側ぎりぎりに描かれる。
EDGE_MARGIN = 0.08


@dataclass
class ChartDetections:
    """1枚の天気図から取り出したもの。位置はすべて相対座標(0〜1)。

    highs / lows は中心の座標。中心が枠外で×が無い場合は、文字の位置を
    edge_highs / edge_lows に入れる(位置として信用しない)。
    """

    highs: list = field(default_factory=list)
    lows: list = field(default_factory=list)
    edge_highs: list = field(default_factory=list)
    edge_lows: list = field(default_factory=list)
    front_segments: dict = field(default_factory=dict)   # 種別 -> Segment のリスト
    stationary_pixels: int = 0


def _mean_or_nan(points: list, axis: int) -> float:
    return float(np.mean([p[axis] for p in points])) if points else float("nan")


def _nearest_distance(a: list, b: list) -> float:
    """2つの点群のあいだの最短距離。片方が空ならNaN。"""
    if not a or not b:
        return float("nan")
    pairs = [
        float(np.hypot(p[0] - q[0], p[1] - q[1]))
        for p in a for q in b
    ]
    return min(pairs)


def _spread(points: list) -> float:
    """点群の広がり(最も離れた2点の距離)。1点以下ならNaN。

    二つ玉低気圧は「低気圧が2つ離れて存在する」という構造なので、
    低気圧の数だけでなく広がりが効くはず。
    """
    if len(points) < 2:
        return float("nan")
    return max(
        float(np.hypot(p[0] - q[0], p[1] - q[1]))
        for i, p in enumerate(points) for q in points[i + 1:]
    )


def _in_both(points: list, region_a, region_b) -> int:
    """2つの領域の**両方**に、それぞれ点があるか。

    領域ごとの在否を別々の特徴量にしていても、木は「両方ある」という組を
    作るのに深さを使う。定義そのものが組であるなら、最初から渡したほうがよい。

    実測では、二つ玉低気圧の手がかりとして `low_in_japan_sea_low` が0.693、
    `low_in_nankigan_low` が0.605あったが、計画が「数えるだけの問題」と
    書いていた `n_low` は0.622に留まった。定義は数ではなく配置である。
    """
    if region_a is None or region_b is None:
        return 0
    return int(
        any(region_a.contains(x, y) for x, y in points)
        and any(region_b.contains(x, y) for x, y in points)
    )


def _on_edge(point: tuple, margin: float = EDGE_MARGIN) -> bool:
    x, y = point
    return x < margin or x > 1 - margin or y < margin or y > 1 - margin


def feature_names(regions: dict) -> list:
    """特徴量の名前を、`build_features` が返す順に並べて返す。"""
    names = [
        "n_high", "n_low", "n_edge_high", "n_edge_low",
        "high_cx", "high_cy", "low_cx", "low_cy",
        "high_low_distance", "low_spread", "high_spread",
        "west_high_east_low", "low_in_japan_sea_and_nankigan",
        "n_warm", "n_cold", "n_occluded",
        "warm_length", "cold_length", "occluded_length",
        "stationary_px",
    ]
    for label in LABELS:
        if label in regions:
            names += [f"high_in_{label}", f"low_in_{label}"]
    return names


def build_features(detections: ChartDetections, regions: dict) -> dict:
    """1枚ぶんの特徴量を辞書で返す。名前は `feature_names` と同じ順。

    地域ごとの高低気圧存在有無は、`src/regions.py` の矩形をそのまま使う。
    計画が「`src/regions.py` は Phase 3 の『地域ごとの高低気圧存在有無』に
    直接使える」と書いていた部分にあたる。
    """
    highs, lows = detections.highs, detections.lows
    segments = detections.front_segments

    def count_frontlike(kind: str) -> int:
        return sum(1 for s in segments.get(kind, []) if s.is_frontlike)

    def total_length(kind: str) -> float:
        return float(sum(s.length for s in segments.get(kind, []) if s.is_frontlike))

    values = {
        "n_high": len(highs),
        "n_low": len(lows),
        "n_edge_high": len(detections.edge_highs),
        "n_edge_low": len(detections.edge_lows),
        "high_cx": _mean_or_nan(highs, 0),
        "high_cy": _mean_or_nan(highs, 1),
        "low_cx": _mean_or_nan(lows, 0),
        "low_cy": _mean_or_nan(lows, 1),
        "high_low_distance": _nearest_distance(highs, lows),
        "low_spread": _spread(lows),
        "high_spread": _spread(highs),
        # 西高東低はラベル名がそのまま配置を表す。負なら高気圧が西、低気圧が東
        "west_high_east_low": (
            _mean_or_nan(highs, 0) - _mean_or_nan(lows, 0)
            if highs and lows else float("nan")
        ),
        # 二つ玉低気圧の定義そのもの。src/labels.py の規約が
        # 「japan_sea_low と nankigan_low を置き換える」と定めている
        "low_in_japan_sea_and_nankigan": _in_both(
            lows, regions.get("japan_sea_low"), regions.get("nankigan_low")),
        "n_warm": count_frontlike("warm_front"),
        "n_cold": count_frontlike("cold_front"),
        "n_occluded": count_frontlike("occluded_front"),
        "warm_length": total_length("warm_front"),
        "cold_length": total_length("cold_front"),
        "occluded_length": total_length("occluded_front"),
        "stationary_px": float(detections.stationary_pixels),
    }
    for label in LABELS:
        region = regions.get(label)
        if region is None:
            continue
        values[f"high_in_{label}"] = int(any(region.contains(x, y) for x, y in highs))
        values[f"low_in_{label}"] = int(any(region.contains(x, y) for x, y in lows))
    return values


# 中心の印と H / L の文字を結びつける距離(相対座標)。実測で決めること。
# `scripts/build_features.py` が処理のあとに距離の分布を出す。
MARK_LETTER_RADIUS = 0.08


def assign_marks_to_letters(marks: list, letters: dict, radius: float = MARK_LETTER_RADIUS):
    """中心の印に、いちばん近い H / L の文字で種別を付ける。

    **印の形では高低を見分けない。**丸で囲んだ×(低気圧)を `circle_cross`
    テンプレートで拾おうとすると失敗する ― 図の側の丸は大きさも太さも
    まちまちなので、丸ごと当てるテンプレートはしきい値を割る一方、丸の内側の
    ×はいつでも完璧に当たる。結果として**低気圧がすべて高気圧として数えられる**
    (実測: 1枚あたり高7.70 / 低0.20)。

    そこで役割を分ける。**位置は印から、種別は文字から取る。**
    ×は中心そのものなので位置として正しく、文字は形が安定していて種別として
    正しい。どちらも得意なほうだけを使う。

    近いものから順に1対1で組にする。返り値は3つ:

    * 種別の付いた印 `{"H": [(cx, cy), ...], "L": [...]}`
    * 組にならなかった文字(中心が枠外の系。×が描かれない)
    * 組にならなかった印(種別が決まらないので位置としては使えない)
    """
    pairs = []
    for i, mark in enumerate(marks):
        for kind, points in letters.items():
            for j, point in enumerate(points):
                distance = math.dist(mark, point)
                if distance <= radius:
                    pairs.append((distance, i, kind, j))
    pairs.sort()

    typed: dict = {kind: [] for kind in letters}
    used_marks: set = set()
    used_letters: set = set()
    for _, i, kind, j in pairs:
        if i in used_marks or (kind, j) in used_letters:
            continue
        used_marks.add(i)
        used_letters.add((kind, j))
        typed[kind].append(marks[i])

    spare_letters = {
        kind: [point for j, point in enumerate(points) if (kind, j) not in used_letters]
        for kind, points in letters.items()
    }
    orphan_marks = [mark for i, mark in enumerate(marks) if i not in used_marks]
    return typed, spare_letters, orphan_marks


def nearest_letter_distances(marks: list, letters: dict) -> list:
    """印ごとに、いちばん近い文字までの距離を返す(半径を決めるための実測用)。

    半径は当てずっぽうで決めてよい数ではない。狭すぎれば種別が付かず、
    広すぎれば隣の系の文字を拾う。
    """
    points = [point for values in letters.values() for point in values]
    if not points:
        return []
    return [min(math.dist(mark, point) for point in points) for mark in marks]


def split_by_edge(points: list, margin: float = EDGE_MARGIN) -> tuple:
    """記号の位置を「中心として信用できるもの」と「縁にあるもの」に分ける。

    中心が枠外の系は×が描かれず文字だけになる。文字の位置は中心ではないので、
    矩形の内外判定に使うと嘘になる。呼ぶ側が×の検出だけを highs/lows に
    入れられるならそちらが正しく、これは文字しか無いときの近似である。
    """
    inside = [p for p in points if not _on_edge(p, margin)]
    edge = [p for p in points if _on_edge(p, margin)]
    return inside, edge


def to_row(detections: ChartDetections, regions: dict) -> list:
    """`feature_names` と同じ順に並べた値のリスト。CSVの1行になる。"""
    values = build_features(detections, regions)
    return [values[name] for name in feature_names(regions)]
