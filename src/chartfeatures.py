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


def _on_edge(point: tuple, margin: float = EDGE_MARGIN) -> bool:
    x, y = point
    return x < margin or x > 1 - margin or y < margin or y > 1 - margin


def feature_names(regions: dict) -> list:
    """特徴量の名前を、`build_features` が返す順に並べて返す。"""
    names = [
        "n_high", "n_low", "n_edge_high", "n_edge_low",
        "high_cx", "high_cy", "low_cx", "low_cy",
        "high_low_distance", "low_spread", "high_spread",
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
