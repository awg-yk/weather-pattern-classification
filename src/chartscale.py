"""天気図の大きさの違いを吸収して、テンプレートを当てられるようにする。

なぜ要るか
----------
`cv2.matchTemplate` は大きさの違いに対応しない。`data/templates` の H/L は
2023年以降の天気図から切り出したものなので、**同じ天気図でも解像度が違うと
1個も当たらない。**国立国会図書館由来の天気図(2000〜2022年)は前処理後の幅が
約1050画素で、2023年以降のものより小さい。

書体も配置も同じ天気図なので、**幅を揃えれば当たるはず**である。実測でも、
記号の大きさは画像の幅に比例していた(3枚の天気図で、数字の高さの中央値が
41・42・43画素、幅に対する割合が 0.0389・0.0392・0.0394 と揃っている)。

やり方
------
テンプレートを切り出した天気図の幅を1つ記録しておき、当てるときは

    倍率 = いま見ている天気図の幅 / 記録した幅

でテンプレートを縮める。座標は相対座標(0〜1)で返しているので、これ以外に
直すところはない。

**ぴったり合わせようとしない。**記録した幅の測り方や、前処理の切り取り位置が
少し動くので、推定した倍率のまわりを少しだけ振って当てる。重なりの抑制
(NMS)が一番よく当たったものを残すので、余計に増えることはない。
"""

import json
from pathlib import Path

# テンプレートを切り出した天気図の幅を書いておくファイル。
# `python -m scripts.set_template_reference` が作る。
REFERENCE_NAME = "reference.json"

# 推定した倍率のまわりを振る幅。前処理の切り取りが数%動いても拾えるようにする。
# 広げすぎると、縮んだテンプレートが等圧線の模様を拾い始める
# (`scripts/diagnose_detection.py` の MAX_PLAUSIBLE 参照)ので、控えめにする。
SPREAD = (0.88, 1.0, 1.13)


def reference_path(templates_dir) -> Path:
    return Path(templates_dir) / REFERENCE_NAME


def load_reference(templates_dir) -> dict | None:
    """記録した基準を読む。無ければ None(自動の倍率は使えない)。"""
    path = reference_path(templates_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    width = data.get("chart_width")
    if not isinstance(width, (int, float)) or width <= 0:
        return None
    return data


def save_reference(templates_dir, chart_width: int, source: str = "") -> Path:
    path = reference_path(templates_dir)
    path.write_text(json.dumps(
        {"chart_width": int(chart_width), "source": source},
        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def auto_letter_size(chart_width: int, templates_dir) -> tuple[float, str]:
    """天気図の幅から、テンプレートを縮める倍率を決める。

    `(倍率, 説明)` を返す。基準が記録されていなければ 1.0 と、その理由を返す。
    """
    reference = load_reference(templates_dir)
    if reference is None:
        return 1.0, (f"{reference_path(templates_dir)} が無いので、"
                     "大きさの自動調整はしません(原寸で当てます)。"
                     "python -m scripts.set_template_reference で作れます")
    ratio = chart_width / float(reference["chart_width"])
    return ratio, (f"幅 {chart_width}px / 基準 {reference['chart_width']}px "
                   f"= 倍率 {ratio:.3f}")


def sizes_around(size: float) -> tuple:
    """推定した倍率のまわりの候補を返す。1.0 のときは振らない。"""
    if size == 1.0:
        return (1.0,)
    return tuple(round(size * s, 4) for s in SPREAD)


def letter_size_arg(value: str):
    """--letter-size の値。数字か "auto"。"""
    if value == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"--letter-size は数字か auto です: {value!r}")
