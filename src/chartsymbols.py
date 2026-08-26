"""天気図から前線と記号を、学習なしで機械的に取り出す。

背景
----
`docs/2026-08-26-detection-plan.md` は、H/L/前線を物体検出してから気圧配置を
判断する方式を提案している。Phase 1・2 はどちらも教師データ(アノテーション)が
要るが、着手前に確かめるべきことがある。**気象庁の JSMAP 天気図は配色が固定で、
記号は同じフォント・同じ大きさで機械的に描画されている**(`scripts/collect_jma.py`
冒頭)。だとすれば、色マスクとテンプレートマッチングだけで前線と記号が取れて
しまい、300枚のアノテーションが要らない可能性がある。

    海岸線・経緯度線=赤茶色、等圧線=黒、温暖前線=赤、寒冷前線=青、閉塞前線=ピンク

このモジュールはその「取れてしまうか」を測るための道具であり、検出器そのもの
ではない。うまくいけば Phase 1・2 の手作業がほぼ不要になり、失敗しても
YOLOの教師データの下地には使える。

閾値は測ってから決める
----------------------
下の `DEFAULT_BANDS` は 2023年1月の8枚を `scripts/chart_palette.py` で
測って決めた(2026-08-26)。分かったこと:

  海岸線・経緯度線  RGB(164, 44, 44)  HSV(  0, 187, 164)  くすんだ赤茶
  寒冷前線          RGB(  4,  4, 252) HSV(120, 251, 252)  純度の高い青
  等圧線            RGB(  4,  4,   4) HSV(  0,   0,   4)  黒

**前線は純度の高い原色、地図の備品はくすんだ色**という分かれ方をしている。
赤茶色と赤は色相がどちらも0で、**色相では分けられない**。分けているのは
明度と彩度である(海岸線 V=164 に対し、原色は V=252)。

温暖前線(純赤)と閉塞前線(ピンク)は、測った8枚が真冬で前線がほとんど無く、
実測できていない。寒冷前線が純青だったことからの類推で帯を置いてある。
夏の停滞前線がある日で測り直して確かめること。

なぜ帯の重なりだけでは足りないか
--------------------------------
`band_overlap()` が0でも安心してはいけない。**片方の帯が空でも0になる**。
実際、最初の暫定値では coastline 帯が空になり、海岸線がまるごと warm_front
に入っていたのに、重なりは0と出た。

そこで `band_variation()` を併せて使う。海岸線・経緯度線は毎回同じ場所に
同じだけ描かれるので画素数がほとんど変わらない(変動係数 0.02)。前線は
日によって有無すら変わる(同 1.30)。**変動の小さい帯は気象ではなく地図の
備品を掴んでいる。**

色空間
------
OpenCVのHSV。色相Hは0〜179(0〜360度の半分)、彩度S・明度Vは0〜255。
赤は色相の原点をまたぐので、`ColorBand` は h_min > h_max のときに巻き戻しと
解釈する(例: h_min=172, h_max=8 は「172〜179 と 0〜8」)。

座標系
------
`src/regions.py` と同じく、画像の左上を(0,0)・右下を(1,1)とする相対座標で
位置を返す。そのまま `Region.contains()` に渡せば「この高気圧はオホーツク海に
あるか」という Phase 3 の特徴量になる。
"""

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

# 前線らしさの下限。これより短い色の塊は、等圧線の縁の色にじみや文字の一部とみなす。
MIN_FRONT_PIXELS = 40

# 前線とみなす細長さ(長軸/短軸)。前線は細長く、凡例や文字は丸い。
MIN_FRONT_ELONGATION = 3.0

# 停滞前線とみなすときの、赤と青が隣り合っていると認める距離(画素)。
STATIONARY_GAP_PX = 12

# 画素数の変動係数がこれ未満なら「毎回同じだけ描かれている」とみなす。
# ただしこれだけでは等圧線も引っかかる(常に図全体を覆うので総量が動かない)。
# 備品かどうかの決め手は FURNITURE_STABILITY のほう。
FURNITURE_CV = 0.15

# 点灯回数の中央値の割合がこれ以上なら、地図の備品とみなす。
# 海岸線・経緯度線は毎回同じ場所にあるので実測で1.000。等圧線は総量こそ
# 動かないが線の位置が日ごとに変わるので、ずっと低くなる。
#
# 「毎回点灯した割合」ではなく中央値を使う理由は MaskAccumulator.stability()
# を参照。上書きで日ごとに違う所が隠れるため、毎回点灯では備品を取り逃がす。
FURNITURE_STABILITY = 0.80


@dataclass(frozen=True)
class ColorBand:
    """HSVの帯で表した1色。h_min > h_max のときは色相の原点をまたぐ。"""

    name: str
    h_min: int
    h_max: int
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """HSV画像から、この帯に入る画素の真偽マスクを返す。"""
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        if self.h_min <= self.h_max:
            in_hue = (h >= self.h_min) & (h <= self.h_max)
        else:  # 赤のように0をまたぐ場合
            in_hue = (h >= self.h_min) | (h <= self.h_max)
        in_sat = (s >= self.s_min) & (s <= self.s_max)
        in_val = (v >= self.v_min) & (v <= self.v_max)
        return in_hue & in_sat & in_val


# 2023年1月の8枚から実測して決めた値(上の docstring を参照)。
# 温暖前線の V>=200 は海岸線(V=164、にじみでも V<=172)と分けるための線引きで、
# 上下に余裕がある。彩度も要求するのは、白へのにじみ(204,132,132 は S=90)を
# 拾わないため。前線の芯だけ取れればよく、にじみは要らない。
DEFAULT_BANDS: dict[str, ColorBand] = {
    "warm_front": ColorBand("warm_front", h_min=172, h_max=8, s_min=200, v_min=200),
    "cold_front": ColorBand("cold_front", h_min=100, h_max=130, s_min=150, v_min=150),
    "occluded_front": ColorBand("occluded_front", h_min=135, h_max=170, s_min=150, v_min=150),
    "coastline": ColorBand("coastline", h_min=170, h_max=20, s_min=120, s_max=210,
                           v_min=120, v_max=199),
    "isobar": ColorBand("isobar", h_min=0, h_max=179, s_max=60, v_max=90),
}

FRONT_BANDS = ("warm_front", "cold_front", "occluded_front")


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB画像(H, W, 3 / uint8)をHSVに変換する。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB画像(H, W, 3)を渡してください: shape={rgb.shape}")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def color_masks(rgb: np.ndarray, bands: dict[str, ColorBand] | None = None) -> dict[str, np.ndarray]:
    """色ごとの真偽マスクをまとめて返す。帯は互いに排他ではない。"""
    bands = bands or DEFAULT_BANDS
    hsv = to_hsv(rgb)
    return {name: band.mask(hsv) for name, band in bands.items()}


def band_overlap(masks: dict[str, np.ndarray]) -> dict[tuple[str, str], int]:
    """帯どうしが同じ画素を取り合っている数を返す。

    温暖前線(赤)と海岸線(赤茶色)の重なりが大きいなら、色だけでは前線を
    切り出せないということであり、閾値を直すか、この方式を諦める根拠になる。
    """
    names = sorted(masks)
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            count = int(np.count_nonzero(masks[a] & masks[b]))
            if count:
                overlap[(a, b)] = count
    return overlap


def band_variation(counts_per_image: dict[str, list[int]]) -> dict[str, dict]:
    """帯ごとの画素数が画像間でどれだけ動くかを返す。地図の備品と気象の切り分け。

    海岸線・経緯度線は毎回同じ場所に同じだけ描かれるので、画素数がほとんど
    変わらない。前線は日によって有無すら変わる。**変動係数が小さい帯は、
    気象ではなく地図の備品を掴んでいる。**

    `band_overlap()` だけでは足りないことがこれで分かる。帯が空でも重なりは
    0になるので、0を「分離できている」と読んではいけない。

    実測(2023年1月の8枚、暫定値のとき):
        warm_front  56322〜60093  変動係数 0.021  -> 海岸線を掴んでいた
        cold_front      0〜 4022  変動係数 1.302  -> 気象を掴んでいた
    """
    result = {}
    for name, counts in counts_per_image.items():
        values = np.asarray(counts, dtype=float)
        mean = float(values.mean()) if values.size else 0.0
        cv = float(values.std() / mean) if mean > 0 else float("nan")
        result[name] = {
            "min": int(values.min()) if values.size else 0,
            "max": int(values.max()) if values.size else 0,
            "mean": mean,
            "cv": cv,
            "looks_like_furniture": bool(mean > 0 and cv < FURNITURE_CV),
        }
    return result


def estimate_shift(reference: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """2枚のマスクの平行移動のずれ (dx, dy, 確からしさ) を位相相関で測る。

    海岸線は毎日まったく同じ地図なので、前処理が揃っていれば ずれは0になる。
    0でないなら `scripts/preprocess_jma.py` のトリミング位置が日ごとに動いて
    いるということで、画像間で座標が比較できない。

    Phase 3 の特徴量は「検出した中心が オホーツク海の矩形に入るか」という
    形で座標を使う(`src/regions.py`)。座標がずれていれば、その判定も
    `data/regions.csv` の矩形も、ずれた分だけ嘘になる。

    返り値の確からしさは0〜1で、1に近いほど平行移動として素直に説明できる。
    """
    if reference.shape != target.shape:
        raise ValueError(
            f"大きさが違うので比べられません: {reference.shape} と {target.shape}。"
            "トリミングの結果が揃っていない可能性がある。"
        )
    a = np.ascontiguousarray(reference, dtype=np.float32)
    b = np.ascontiguousarray(target, dtype=np.float32)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    return float(dx), float(dy), float(response)


def label_auc(values: list[float], has_label: list[bool]) -> float:
    """ラベルの有無を測定値だけでどれだけ分けられるかを返す(ROC-AUC)。

    しきい値を決めずに順位だけで測るので、値の単位に依らない。
    0.5 が当てずっぽう、1.0 が完全な分離。同じ値は引き分けとして0.5で数える。

    しきい値最適化つきの本評価は `src/evaluate.py` の仕事で、これはその前の
    「そもそも信号があるか」を見るための道具である。**この数字を本評価の
    代わりに報告してはいけない。**
    """
    positives = [v for v, flag in zip(values, has_label) if flag]
    negatives = [v for v, flag in zip(values, has_label) if not flag]
    if not positives or not negatives:
        return float("nan")
    wins = sum(
        (p > n) + 0.5 * (p == n)
        for p in positives
        for n in negatives
    )
    return wins / (len(positives) * len(negatives))


class MaskAccumulator:
    """帯ごとに「各画素が何枚の画像で点灯したか」を数える。

    `band_variation()` の画素数だけでは等圧線と海岸線を分けられない。等圧線は
    常に図全体を覆うので総量が動かず、備品と同じ顔をする。**位置**を見れば
    分かれる ― 海岸線は毎回まったく同じ画素に描かれ、等圧線は日ごとに動く。

    画像の大きさが違うと足し合わせられないので、その場合は諦めて報告する
    (実測では2023年1月が1453x1500、2024年7月が1453x1499で1画素違った)。
    """

    def __init__(self):
        self.counts: dict[str, np.ndarray] = {}
        self.n_images = 0
        self.shape: tuple | None = None
        self.size_mismatch = False

    def add(self, masks: dict[str, np.ndarray]) -> None:
        shape = next(iter(masks.values())).shape
        if self.shape is None:
            self.shape = shape
        elif shape != self.shape:
            self.size_mismatch = True
            return
        for name, mask in masks.items():
            if name not in self.counts:
                self.counts[name] = np.zeros(shape, dtype=np.int32)
            self.counts[name] += mask.astype(np.int32)
        self.n_images += 1

    def stability(self) -> dict[str, dict]:
        """帯の画素がどれだけ動かないかを2つの数で返す。

        typical : 点灯回数の中央値 ÷ 枚数。**判定に使うのはこちら。**
        always  : 毎回点灯した画素 ÷ 一度でも点灯した画素。参考値。

        always を判定に使うと備品を取り逃がす。海岸線は毎日同じ場所にあるが、
        等圧線や前線がその上に描かれて**日ごとに違う所が隠れる**。実測では
        1枚あたり5.1%が隠れており、12枚では 0.949^12 = 0.53 まで落ちた。
        毎日そこにあるのに「毎回点灯」しないので、always は0.533になる。

        中央値なら隠れは効かない。ある画素が12枚中10枚で見えていれば
        中央値は動かない。実測では海岸線 1.000、等圧線・前線はずっと低い。
        """
        result = {}
        for name, counts in self.counts.items():
            ever_mask = counts >= 1
            ever = int(np.count_nonzero(ever_mask))
            if not ever:
                result[name] = {"typical": float("nan"), "always": float("nan")}
                continue
            always = int(np.count_nonzero(counts == self.n_images))
            median_k = float(np.median(counts[ever_mask]))
            result[name] = {
                "typical": median_k / self.n_images,
                "always": always / ever,
            }
        return result


def clean_mask(mask: np.ndarray, min_pixels: int = MIN_FRONT_PIXELS) -> np.ndarray:
    """小さすぎる塊を落とす。JPEG由来の色にじみと、文字の一部を除くため。"""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_pixels
    return keep[labels]


@dataclass(frozen=True)
class Segment:
    """色マスクの連結成分ひとつ。前線の一区間にあたる。"""

    kind: str
    pixels: int
    length: float        # 長軸方向の広がり(画素)
    elongation: float    # 長軸/短軸。前線は細長く、文字や凡例は丸い
    cx: float            # 相対座標(0〜1)の重心
    cy: float

    @property
    def is_frontlike(self) -> bool:
        return self.elongation >= MIN_FRONT_ELONGATION


def _pca_shape(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float]:
    """画素の散らばりから (長軸の広がり, 細長さ) を返す。"""
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64)])
    coords = coords - coords.mean(axis=1, keepdims=True)
    if coords.shape[1] < 2:
        return 0.0, 1.0
    cov = np.cov(coords)
    eigenvalues = np.linalg.eigvalsh(cov)
    minor, major = float(max(eigenvalues[0], 0.0)), float(max(eigenvalues[1], 0.0))
    major_std, minor_std = np.sqrt(major), np.sqrt(minor)
    # 標準偏差そのものではなく、一様な線分の長さに直す(一様分布の標準偏差は長さ/√12)
    length = major_std * np.sqrt(12.0)
    elongation = major_std / minor_std if minor_std > 1e-6 else float("inf")
    return length, elongation


def segments(mask: np.ndarray, kind: str, min_pixels: int = MIN_FRONT_PIXELS) -> list[Segment]:
    """マスクを連結成分に分け、前線らしさを測って返す。"""
    height, width = mask.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    found = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        ys, xs = np.nonzero(labels == i)
        length, elongation = _pca_shape(ys, xs)
        found.append(Segment(
            kind=kind,
            pixels=area,
            length=length,
            elongation=elongation,
            cx=float(centroids[i][0]) / width,
            cy=float(centroids[i][1]) / height,
        ))
    return sorted(found, key=lambda s: s.pixels, reverse=True)


def stationary_mask(
    warm: np.ndarray, cold: np.ndarray, gap_px: int = STATIONARY_GAP_PX
) -> np.ndarray:
    """赤と青が交互に並ぶ区間(停滞前線)のマスクを返す。

    停滞前線は同じ線の上で赤と青が交互に描かれる。それぞれを gap_px だけ
    太らせて重なりを取れば、「赤の隣に青がある」場所だけが残る。温暖前線
    (赤だけ)・寒冷前線(青だけ)は、離れて描かれているかぎり残らない。
    """
    kernel = np.ones((2 * gap_px + 1, 2 * gap_px + 1), np.uint8)
    warm_grown = cv2.dilate(warm.astype(np.uint8), kernel)
    cold_grown = cv2.dilate(cold.astype(np.uint8), kernel)
    both = (warm_grown & cold_grown).astype(bool)
    return both & (warm | cold)


@dataclass(frozen=True)
class Candidate:
    """記号(H/L/T/TD や中心の×)かもしれない小さな黒い塊。"""

    x0: int
    y0: int
    x1: int
    y1: int
    pixels: int
    fill: float          # bboxに占める画素の割合
    cx: float            # 相対座標(0〜1)
    cy: float
    label: str = ""      # テンプレートマッチングで付いた名前
    score: float = 0.0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def glyph_candidates(
    rgb: np.ndarray,
    bands: dict[str, ColorBand] | None = None,
    min_side: int = 6,
    max_side: int = 64,
    min_fill: float = 0.12,
) -> list[Candidate]:
    """記号の候補になる小さな孤立した黒い塊を、テンプレート無しで拾う。

    等圧線は図の端から端まで繋がった1つの巨大な連結成分になるので、bboxの
    一辺で足切りするだけで落ちる。残るのは記号(H/L/T/TD・×)と、気圧の数値・
    緯度経度の目盛といった文字である。**この段階では両者を区別しない。**
    区別はテンプレートマッチング(`match_templates`)の仕事で、ここが返すのは
    「1枚あたり何個を相手にすればよいか」という規模の見積もりになる。

    min_side / max_side は 1453x1500 の気象庁PDF版で測った値に合わせてある
    (記号は28〜32画素に集まり、大きいもので43x46)。前処理の拡大率が違う
    画像を扱うときは、`scripts/extract_symbols.py scan --max-side` で広げて、
    拾える個数が変わるかを見ること。
    """
    bands = bands or DEFAULT_BANDS
    height, width = rgb.shape[:2]
    black = bands["isobar"].mask(to_hsv(rgb))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        black.astype(np.uint8), connectivity=8
    )
    found = []
    for i in range(1, count):
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        fill = area / float(w * h)
        if fill < min_fill:
            continue
        found.append(Candidate(
            x0=x, y0=y, x1=x + w, y1=y + h,
            pixels=area, fill=fill,
            cx=float(centroids[i][0]) / width,
            cy=float(centroids[i][1]) / height,
        ))
    return sorted(found, key=lambda c: (c.y0, c.x0))


def match_templates(
    rgb: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = 0.8,
    bands: dict[str, ColorBand] | None = None,
) -> list[Candidate]:
    """記号のテンプレートを画像全体に当て、閾値を超えた位置を返す。

    テンプレートは2値(記号の画素がTrue)で渡す。天気図側も同じ2値マスクに
    してから当てるので、JPEGの圧縮ノイズや紙の地色に左右されにくい。
    同じ場所に複数当たったときは、スコアの高いものだけを残す(NMS)。
    """
    bands = bands or DEFAULT_BANDS
    height, width = rgb.shape[:2]
    black = bands["isobar"].mask(to_hsv(rgb)).astype(np.float32)
    hits: list[Candidate] = []
    for name, template in templates.items():
        patch = template.astype(np.float32)
        if patch.shape[0] > black.shape[0] or patch.shape[1] > black.shape[1]:
            continue
        response = cv2.matchTemplate(black, patch, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.nonzero(response >= threshold)
        h, w = patch.shape
        for y, x in zip(ys, xs):
            hits.append(Candidate(
                x0=int(x), y0=int(y), x1=int(x) + w, y1=int(y) + h,
                pixels=int(patch.sum()), fill=float(patch.mean()),
                cx=(x + w / 2) / width, cy=(y + h / 2) / height,
                label=name, score=float(response[y, x]),
            ))
    return _suppress_overlaps(hits)


def _suppress_overlaps(hits: list[Candidate], max_overlap: float = 0.2) -> list[Candidate]:
    """重なった検出のうちスコアが最大のものだけ残す。

    重なりは IoU ではなく「小さいほうの面積に対する重なりの割合」で測る。
    記号は互いに入れ子になる ― "H" の右の縦棒は "L" に似ており、"TD" の中には
    "T" がある。入れ子は IoU では小さく出るので、IoU で抑えると内側の誤検出が
    生き残る。小さいほうを基準にすれば、内側にすっぽり入った検出は必ず落ちる。

    別々の記号どうしは天気図の上で重ならない(重なりの割合は0)ので、閾値は
    低めでよい。ただし主たる防波堤は `match_templates` の threshold のほうで、
    抑制はその取りこぼしを拾う二段目にすぎない。
    """
    kept: list[Candidate] = []
    for hit in sorted(hits, key=lambda c: c.score, reverse=True):
        if all(_overlap_ratio(hit, other) <= max_overlap for other in kept):
            kept.append(hit)
    return sorted(kept, key=lambda c: (c.y0, c.x0))


def _overlap_ratio(a: Candidate, b: Candidate) -> float:
    """重なった面積 / 小さいほうの面積。入れ子なら1.0に近づく。"""
    inter_w = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    inter_h = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    smaller = min(a.width * a.height, b.width * b.height)
    return inter / smaller if smaller else 0.0


def crop_template(rgb: np.ndarray, box: tuple[int, int, int, int],
                  bands: dict[str, ColorBand] | None = None) -> np.ndarray:
    """天気図の一部を切り出して2値テンプレートにする。

    テンプレートは本物の天気図から採るしかない(フォントが手元に無い)。
    記号を1つ見つけて座標を渡せば、以降はそれを全画像に当てられる。
    """
    bands = bands or DEFAULT_BANDS
    x0, y0, x1, y1 = box
    black = bands["isobar"].mask(to_hsv(rgb))
    return black[y0:y1, x0:x1]


def dominant_colors(rgb: np.ndarray, top: int = 12, ignore_near_white: int = 235) -> list[dict]:
    """画像に多い色を、画素数の多い順に返す。閾値を決めるための実測用。

    紙の地色(ほぼ白)は数が桁違いに多くて他を押し流すので除く。
    量子化して数えるのは、JPEGの圧縮で1色が微妙に散るため。
    """
    flat = rgb.reshape(-1, 3)
    keep = flat.min(axis=1) < ignore_near_white
    flat = flat[keep]
    if flat.size == 0:
        return []
    quantized = (flat // 8) * 8 + 4
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1][:top]
    total = int(counts.sum())
    result = []
    for i in order:
        color = colors[i].astype(np.uint8)
        hsv = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV).reshape(3)
        result.append({
            "rgb": tuple(int(v) for v in color),
            "hsv": tuple(int(v) for v in hsv),
            "pixels": int(counts[i]),
            "share": float(counts[i]) / total,
        })
    return result
