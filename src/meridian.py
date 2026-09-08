"""記号の傾きを、天気図上の位置から予測するための部品。

背景
----
気象庁の天気図では、高気圧・低気圧の H / L は**経線と平行に描かれている**。
経線は図の上で垂直ではないので、記号は場所ごとに傾く。実測では
`data/templates` の12枚が -36.5度 〜 +22.0度 に散らばっていた。

テンプレートマッチングはこの傾きを、テンプレートの側を -60度から +60度まで
5度刻みで回して吸収している。刻みの真ん中(2.5度ずれ)に落ちた記号は
スコアが 0.935 から 0.705 まで落ち、そこに等圧線が1本重なると 0.605 で
しきい値を割る。**同じ天気図の中で当たり外れが出る主な理由がこれである**
(`docs/2026-09-08-symbol-tilt-and-meridians.md`)。

考え方
------
経線は投影法によらず**図の外の1点(極、円錐図法なら円錐の頂点)から放射状に
出る直線**になる。したがって点 (x, y) での経線の向きは、その1点から
(x, y) へ向かう向きに等しい。**点を2つ決めるだけで、画像上のどの画素でも
期待される傾きが計算できる**ということである。

この点の座標は緯度経度から求めない。`src/sites.py` と同じ理由で、前処理の
autocrop_to_content() が白縁を落とすため画素と緯度経度の対応表が無い。
**検出できている記号の(位置, 傾き)から当てる**(`scripts/fit_meridian.py`)。

座標系と符号
------------
位置は `src/regions.py` `src/sites.py` と同じ相対座標(左上が(0,0)、
右下が(1,1))。傾きは `src.chartsymbols.rotate_template` に渡す角度と同じ
符号で、度で表す。テンプレート自身の傾きを t、当たった回転角を a とすると、
**その記号の傾きは t + a** になる(12枚すべてで2度以内に一致することを
確認済み)。
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MERIDIAN_PATH = Path(__file__).resolve().parent.parent / "data" / "meridian.json"

# 傾きを測るときに振る範囲と刻み。テンプレートの実測が -36.5〜+22.0度なので
# 60度あれば足りる。0.5度刻みにしているのは、当てはめの残差を1度未満で
# 見たいため(照合の刻みとは別物)。
TILT_RANGE = 60.0
TILT_STEP = 0.5


def template_tilt(mask: np.ndarray, span: float = TILT_RANGE,
                  step: float = TILT_STEP) -> float:
    """2値のテンプレートが持っている傾きを度で返す。

    H も L も縦棒を持つので、**まっすぐに戻したときに縦の画素が列にそろう**。
    列ごとの画素数を確率とみなし、その二乗和(とがり具合)が最大になる向きを
    探す。回転で画布が広がっても、空の列は確率0で寄与しないので効かない。

    妥当性は相互照合で確かめてある。記号 X を置いてテンプレート Y で当てると
    当たる角度は tilt(X) - tilt(Y) になるはずで、12組すべてが2度以内で
    一致した(`docs/2026-09-08-symbol-tilt-and-meridians.md`)。
    """
    from src.chartsymbols import rotate_template

    best_score, best_angle = -1.0, 0.0
    for angle in np.arange(-span, span + step / 2, step):
        rotated = rotate_template(mask, float(angle))
        columns = rotated.sum(axis=0).astype(np.float64)
        total = columns.sum()
        if total == 0:
            continue
        share = columns / total
        score = float((share ** 2).sum())
        if score > best_score:
            best_score, best_angle = score, float(angle)
    # まっすぐに戻すのに best_angle だけ回したのだから、記号の傾きはその逆向き
    return -best_angle


def template_tilts(templates: dict[str, np.ndarray]) -> dict[str, float]:
    """テンプレート一式の傾きをまとめて測る。"""
    return {name: template_tilt(mask) for name, mask in templates.items()}


@dataclass(frozen=True)
class Meridian:
    """経線が集まる1点。位置から記号の傾きを予測する。

    `x` `y` は相対座標で、**画像の外に出る**(日本周辺の天気図なら上のほう)。
    `sign` は `rotate_template` の回転の向きと図の向きの食い違いを吸収する
    ためのもので、当てはめのときに +1 / -1 の良いほうを選ぶ。
    `width` `height` は当てはめに使った画像の画素数で、縦横比が違う画像に
    そのまま使うと角度がずれるため記録しておく。
    """

    x: float
    y: float
    sign: float = 1.0
    width: int = 1453
    height: int = 1500
    residual: float = float("nan")   # 当てはめの残差(度、中央絶対偏差)
    samples: int = 0
    note: str = ""

    def tilt_at(self, x, y):
        """相対座標 (x, y) での記号の傾きを度で返す。配列でも渡せる。"""
        dx = (np.asarray(x, dtype=float) - self.x) * self.width
        dy = (np.asarray(y, dtype=float) - self.y) * self.height
        return self.sign * np.degrees(np.arctan2(dx, dy))

    def deviation(self, x, y, tilt):
        """観測した傾きが予測から何度ずれているかを返す。"""
        return _wrap(np.asarray(tilt, dtype=float) - self.tilt_at(x, y))

    def angles_for(self, x, y, template_tilt_deg: float,
                   span: float = 3.0, step: float = 0.5) -> list[float]:
        """その位置でそのテンプレートに必要な回転角を、細かい刻みで返す。

        記号の傾き = テンプレートの傾き + 回転角 なので、必要な回転角は
        予測した傾きからテンプレートの傾きを引いたものになる。
        """
        centre = float(self.tilt_at(x, y)) - template_tilt_deg
        n = int(round(span / step))
        return [centre + i * step for i in range(-n, n + 1)]

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "sign": self.sign,
                "width": self.width, "height": self.height,
                "residual": self.residual, "samples": self.samples,
                "note": self.note}

    def save(self, path=DEFAULT_MERIDIAN_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False,
                                         indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path=DEFAULT_MERIDIAN_PATH) -> "Meridian":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _wrap(degrees: np.ndarray) -> np.ndarray:
    """角度の差を -90度〜+90度に畳む。記号の傾きに180度の折り返しは無いが、
    当てはめの途中で極を通り越すと符号が飛ぶので、そこだけ吸収する。"""
    return (degrees + 90.0) % 180.0 - 90.0


def fit(xs, ys, tilts, width: int = 1453, height: int = 1500,
        note: str = "") -> Meridian:
    """(位置, 傾き)の並びから、経線が集まる点を当てる。

    未知数は点の座標2つだけ。**外れ値が必ず混じる**(誤検出、等圧線に
    削られた記号)ので、二乗和ではなく Huber 損失で当てる。残差は中央絶対
    偏差で報告する ― 平均だと外れ値1個で見え方が変わってしまう。
    """
    from scipy.optimize import least_squares

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    tilts = np.asarray(tilts, dtype=float)
    if xs.size < 3:
        raise ValueError(f"当てはめには3個以上の標本が要ります: {xs.size}個")

    best = None
    for sign in (1.0, -1.0):
        def residuals(p, sign=sign):
            trial = Meridian(x=p[0], y=p[1], sign=sign, width=width, height=height)
            return trial.deviation(xs, ys, tilts)

        # 出発点は「図の真上のはるか遠く」。日本周辺の天気図なら極は上にある
        for start in ((0.5, -1.0), (0.5, -3.0), (0.5, 2.0)):
            try:
                out = least_squares(residuals, x0=np.array(start),
                                    loss="huber", f_scale=5.0)
            except Exception:
                continue
            got = Meridian(x=float(out.x[0]), y=float(out.x[1]), sign=sign,
                           width=width, height=height)
            spread = float(np.median(np.abs(got.deviation(xs, ys, tilts))))
            if best is None or spread < best[0]:
                best = (spread, got)

    if best is None:
        raise RuntimeError("当てはめが1通りも収束しませんでした")
    spread, got = best
    return Meridian(x=got.x, y=got.y, sign=got.sign, width=width, height=height,
                    residual=spread, samples=int(xs.size), note=note)


# 粗い探索で拾うしきい値。**本来のしきい値より低くしてある。**5度刻みの
# 探索では、刻みの真ん中に落ちた記号は 0.505 まで下がりうる(実測)。
# ここで拾っておかないと、測り直す機会そのものが無くなる。
COARSE_THRESHOLD = 0.45

# 予測からこれ以上ずれた傾きで当たったものは誤検出として捨てる。
# テンプレート12枚の当てはめ残差が2度以内であることを見込み、粗い探索の
# 刻み(5度)と当てはめの誤差に余裕を持たせた値。
MAX_DEVIATION = 12.0


def refine_hits(rgb: np.ndarray, hits: list, templates: dict[str, np.ndarray],
                tilts: dict[str, float], meridian: Meridian, *,
                threshold: float, bands=None,
                span: float = 3.0, step: float = 0.5,
                max_deviation: float = MAX_DEVIATION,
                slack: int = 6) -> list:
    """粗く拾った候補を、予測した角度で測り直して採否を決める。

    やることは2つだけ。

    1. **予測とかけ離れた傾きで当たったものを捨てる。**その場所の経線が
       -10度なのに +40度で当たった、というのは記号ではない見込みが高い。
    2. **残ったものを、予測した角度の前後だけ細かく振って測り直す。**
       画像全体ではなく候補の周りだけを見るので、刻みを0.5度にしても
       ほとんど時間がかからない。

    測り直しは `sizes=(1.0,)` の検出を前提にしている(本番の設定)。倍率を
    振って拾った候補は、テンプレートの大きさが分からないので測り直せない。
    """
    from src.chartsymbols import (DEFAULT_BANDS, _suppress_overlaps, rotate_template,
                                  to_hsv, Candidate)

    bands = bands or DEFAULT_BANDS
    height, width = rgb.shape[:2]
    ink = bands["isobar"].mask(to_hsv(rgb)).astype(np.float32)

    kept: list = []
    for hit in hits:
        template = templates.get(hit.label)
        if template is None:
            continue
        observed = tilts[hit.label] + hit.angle
        if abs(float(meridian.deviation(hit.cx, hit.cy, observed))) > max_deviation:
            continue

        patches = []
        for angle in meridian.angles_for(hit.cx, hit.cy, tilts[hit.label], span, step):
            patch = rotate_template(template, float(angle)).astype(np.float32)
            if patch.sum() > 0:
                patches.append((float(angle), patch))
        if not patches:
            continue

        # 候補の周りだけを切り出す。いちばん大きく回った型が入る余裕を取る
        reach = max(max(p.shape) for _, p in patches) // 2 + slack
        cx_px = (hit.x0 + hit.x1) // 2
        cy_px = (hit.y0 + hit.y1) // 2
        wx0, wy0 = max(0, cx_px - reach), max(0, cy_px - reach)
        wx1, wy1 = min(width, cx_px + reach), min(height, cy_px + reach)
        window = ink[wy0:wy1, wx0:wx1]

        best = None
        for angle, patch in patches:
            if patch.shape[0] > window.shape[0] or patch.shape[1] > window.shape[1]:
                continue
            response = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(response)
            if best is None or score > best[0]:
                best = (float(score), angle, loc, patch.shape)
        if best is None or best[0] < threshold:
            continue

        score, angle, (rx, ry), (ph, pw) = best
        x0, y0 = wx0 + int(rx), wy0 + int(ry)
        kept.append(Candidate(
            x0=x0, y0=y0, x1=x0 + pw, y1=y0 + ph,
            pixels=hit.pixels, fill=hit.fill,
            cx=(x0 + pw / 2) / width, cy=(y0 + ph / 2) / height,
            label=hit.label, score=score, angle=angle,
        ))
    return _suppress_overlaps(kept)
