"""確信度(表示する%)を、実際に当たる割合に合わせて校正する。

■ なぜ必要か

モデルの生の出力 sigmoid(logit) をそのまま「確信度」として見せると、
実際の的中率よりかなり高い数字が出る。原因は2つある。

1) 学習時の pos_weight による系統的なかさ上げ
   src/train.py は BCEWithLogitsLoss(pos_weight=w) で学習している。これは
   「陽性を見逃したときの損失」をw倍する重み付けで、少数ラベルの取りこぼしを
   防ぐために入れてある。ただしこの損失を最小にする出力は、真の確率 p に対して

       sigmoid(z) = w·p / (w·p + (1 - p))

   になる。つまり出力は構造的に p より大きくなる。逆に解くと

       p = sigmoid(z - log w)

   なので、ロジットから log w を引けば元の確率に戻せる。
   pos_weight=8 のラベルなら、表示60%の中身は
   p = sigmoid(logit(0.6) - log 8) ≒ 0.16 でしかない。
   「人が見れば明らかに違うのに確信度60%」の大部分はこれで説明がつく。
   しかも w はラベルごとに違うので、かさ上げの量もラベルごとに違い、
   ラベル間で確信度を比べること自体が成り立たなくなっている。

2) 学習データが少ないCNN自体の自信過剰
   数百〜数千枚規模だと、モデルは訓練データを覚えてしまい、出力が0/1に寄る。
   pos_weightを取り除いてもなお、表示90%の的中率が70%といったズレが残る。

■ どう直すか

検証データを使い、ラベルごとに1次元のロジスティック回帰を当てはめ直す
(Platt scaling)。

    p_calibrated = sigmoid(a · z + b)

a=1, b=-log w が上の1)の補正そのものなので、この形は pos_weight の補正を
含んでいる。a が1より小さく出れば、それは2)の自信過剰をならしたぶん。

検証データに陽性がほとんど無いラベルは a, b を当てはめられない(過学習する)ので、
その場合は 1) の解析的な補正だけを使う(method="prior")。

■ この方法で消えるもの・消えないもの

消える : 「実際には16%しかないものを60%と表示する」という表示上のかさ上げ。
         校正後は、確信度60%と出た事例のうち実際に約6割が当たる、という意味になる。
消えない: モデルが本当に自信満々で間違えるケース。これは校正では減らせない。
         ただし校正後は件数として数えられる(reliability_table を参照)ので、
         どのくらい残っているかが分かる。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.labels import LABELS

# 検証データにこの数だけ陽性が無いラベルは、Platt scaling を当てはめず
# pos_weight の解析的な補正だけを使う。少数の点に当てはめた a, b は
# 検証データの偶然に引っ張られ、かえって表示が歪むため。
MIN_POSITIVES_FOR_FIT = 10

# 当てはめ時の正則化の強さ。(log a)^2 + (b - b0)^2 に掛ける。
# 損失は合計(平均ではない)で測るので、データが多いほど相対的に効かなくなる。
FIT_REGULARIZATION = 1.0

# 校正ファイルの書式が変わったときに、古いファイルを見分けるための版番号
CALIBRATION_VERSION = 1


def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + exp(x)) を、xが大きくてもあふれない形で計算する。"""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """ロジットを確率に直す。指数のあふれを避けるため符号で場合分けする。"""
    x = np.asarray(x, dtype="float64")
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """確率をロジットに戻す。0や1ちょうどが来ても落ちないよう内側に丸める。"""
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def remove_class_weight_bias(probs: np.ndarray, weight_ratio: np.ndarray) -> np.ndarray:
    """陽性側を weight_ratio 倍に重み付けして学習したモデルの確率を、元に戻す。

    重み付き学習が出す確率 q と、重み無しなら出たはずの確率 p の間には
        q = w·p / (w·p + (1 - p))   すなわち   p = sigmoid(logit(q) - log w)
    の関係がある。CNN側の pos_weight でも、scikit-learnの
    class_weight="balanced"(w = 陰性数/陽性数)でも同じ式が成り立つ。
    """
    return sigmoid(logit(probs) - np.log(np.maximum(weight_ratio, 1e-12)))


def fit_platt(
    logits: np.ndarray,
    targets: np.ndarray,
    b_init: float = 0.0,
    regularization: float = FIT_REGULARIZATION,
) -> tuple[float, float]:
    """1ラベル分の (a, b) を、負の対数尤度が最小になるように当てはめる。

    logits  : そのラベルの生のロジット (n,)
    targets : 0/1の正解 (n,)
    b_init  : bの初期値かつ正則化の中心。pos_weight が分かっていれば -log w を渡す。
              これにより「データが足りない方向にはpos_weightの解析解へ寄る」挙動になる。

    a > 0 を保つため u = log a で最適化する(aが負だと確率の大小が逆転してしまう)。
    """
    from scipy.optimize import minimize

    z = np.asarray(logits, dtype="float64")
    y = np.asarray(targets, dtype="float64")

    def objective(params):
        u, b = params
        a = np.exp(u)
        s = a * z + b
        # NLL = Σ softplus(s) - y·s  (数値的に安定な二値交差エントロピー)
        nll = float(np.sum(_softplus(s) - y * s))
        grad_s = sigmoid(s) - y
        g_u = float(np.sum(grad_s * z) * a)
        g_b = float(np.sum(grad_s))
        # 正則化: データが少ないとき a=1, b=b_init に引き戻す
        nll += regularization * (u ** 2 + (b - b_init) ** 2)
        g_u += 2.0 * regularization * u
        g_b += 2.0 * regularization * (b - b_init)
        return nll, np.array([g_u, g_b])

    result = minimize(objective, x0=np.array([0.0, b_init]), jac=True, method="L-BFGS-B")
    u, b = result.x
    return float(np.exp(u)), float(b)


def find_best_threshold(probs: np.ndarray, targets: np.ndarray, steps: int = 97) -> float:
    """1ラベル分の、F1が最大になる判定しきい値を探す。

    校正後の確率は少数ラベルほど小さい値に収まるため、src/evaluate.py の
    0.05刻み(下限0.05)では粗すぎる。ここでは0.01刻みで探す。

    下限を0.02に置くのは、頻度の高いラベルでF1が「常に陽性」で最大になり、
    しきい値が0近くに落ちるのを避けるため。そうなったラベルは事実上いつでも
    予測に出てしまい、確信度を見て絞り込むという運用ができなくなる。
    """
    from sklearn.metrics import f1_score

    if targets.sum() == 0:
        return 0.5
    candidates = np.linspace(0.02, 0.98, steps)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(targets, (probs > t).astype(float), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


@dataclass
class LabelCalibration:
    """1ラベル分の校正パラメータ。"""

    a: float = 1.0
    b: float = 0.0
    threshold: float = 0.5
    # "platt" = 検証データに当てはめた / "prior" = pos_weightの解析的な補正のみ /
    # "none"  = 補正なし(pos_weightも不明で、検証データにも陽性が無い)
    method: str = "none"
    n_positive: int = 0

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * np.asarray(logits, dtype="float64") + self.b)


@dataclass
class Calibration:
    """全ラベル分の校正。JSONで保存し、推論時に読み込んで使う。"""

    per_label: dict = field(default_factory=dict)
    labels: list = field(default_factory=lambda: list(LABELS))
    source: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    version: int = CALIBRATION_VERSION

    # ---- 適用 ----

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        """生のロジット (n, L) または (L,) を校正済み確率にする。"""
        z = np.asarray(logits, dtype="float64")
        single = z.ndim == 1
        z = np.atleast_2d(z)
        out = np.empty_like(z)
        for i, label in enumerate(self.labels):
            out[:, i] = self[label].apply(z[:, i])
        return out[0] if single else out

    def from_probabilities(self, probs: np.ndarray) -> np.ndarray:
        """すでに sigmoid を通した確率しか手元に無い場合の入り口。

        sigmoid は可逆なのでロジットに戻してから校正する。結果は
        probabilities() にロジットを渡した場合と一致する。
        """
        return self.probabilities(logit(probs))

    def thresholds(self) -> np.ndarray:
        return np.array([self[label].threshold for label in self.labels])

    def predicted_labels(self, probs: np.ndarray) -> list:
        """校正済み確率から、しきい値を超えたラベルを返す。"""
        probs = np.asarray(probs, dtype="float64").ravel()
        return [label for i, label in enumerate(self.labels)
                if probs[i] > self[label].threshold]

    def is_confident(self, label: str, prob: float) -> bool:
        """そのラベルの確率が、判定しきい値を超えているか。"""
        return float(prob) > self[label].threshold

    def __getitem__(self, label: str) -> LabelCalibration:
        return self.per_label.get(label, LabelCalibration())

    @property
    def is_fitted(self) -> bool:
        return any(c.method != "none" for c in self.per_label.values())

    # ---- 保存・読み込み ----

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "labels": list(self.labels),
            "source": self.source,
            "metrics": self.metrics,
            "per_label": {k: asdict(v) for k, v in self.per_label.items()},
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def from_dict(cls, payload: dict) -> "Calibration":
        labels = payload.get("labels", list(LABELS))
        if labels != list(LABELS):
            raise ValueError(
                "校正ファイルのラベル構成が現在のsrc/labels.pyと一致しません。\n"
                f"  ファイル側: {labels}\n"
                f"  現在      : {list(LABELS)}\n"
                "scripts/calibrate.py で作り直してください。"
            )
        return cls(
            per_label={k: LabelCalibration(**v) for k, v in payload.get("per_label", {}).items()},
            labels=labels,
            source=payload.get("source", {}),
            metrics=payload.get("metrics", {}),
            version=payload.get("version", CALIBRATION_VERSION),
        )

    @classmethod
    def load(cls, path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def average(cls, calibrations) -> "Calibration":
        """複数の校正を、ラベルごとにパラメータの平均で1つにまとめる。

        交差検証の各foldで作った校正を、アンサンブルに1つだけ適用したいときに使う。

        注意: これは近似である。厳密には「3モデルの平均出力」に対する校正を、
        その平均出力を測った検証データから当てはめ直すべきだが、LOYOでは
        foldごとに検証データが違うため、アンサンブル共通の検証データが存在しない。
        個々のfoldの係数を平均するのは、その代わりの実務的な妥協。
        """
        calibrations = list(calibrations)
        if not calibrations:
            return cls.identity()
        per_label = {}
        for label in LABELS:
            parts = [c[label] for c in calibrations]
            per_label[label] = LabelCalibration(
                a=float(np.mean([p.a for p in parts])),
                b=float(np.mean([p.b for p in parts])),
                threshold=float(np.mean([p.threshold for p in parts])),
                method="average",
                n_positive=int(sum(p.n_positive for p in parts)),
            )
        return cls(per_label=per_label)

    @classmethod
    def identity(cls) -> "Calibration":
        """校正なし。生の出力をそのまま返し、しきい値は一律0.5。"""
        return cls(per_label={label: LabelCalibration() for label in LABELS})


def file_fingerprint(path, chunk_size: int = 1 << 20) -> str:
    """ファイルの内容のSHA-256(先頭16桁)。校正が「どの重み・どのラベルで作られたか」の記録用。

    パスや更新時刻ではなく中身を見る。重みを学習し直すと必ず変わるので、
    古い校正ファイルが新しい重みの隣に残っていても検出できる。
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def default_path(weights_path) -> Path:
    """重みファイルに対応する校正ファイルの既定の置き場所。

    weights/model.pt -> weights/model.calib.json
    重みごとに校正が変わる(fold違いの重みは別の校正が要る)ため、
    重みのパスから機械的に決める。
    """
    path = Path(weights_path)
    return path.with_suffix(".calib.json")


class StaleCalibrationError(RuntimeError):
    """校正ファイルが、隣にある重みとは別の重みから作られている。"""


def load_for_weights(weights_path, verbose: bool = True) -> Calibration:
    """重みに対応する校正ファイルがあれば読む。無ければ校正なしを返す。

    校正ファイルが無い状態でも、これまで通り動くことを優先する
    (ただし表示は「未校正」と分かるようにする)。

    校正は重みごとに違うので、重みを学習し直したら作り直さなければならない。
    古い校正ファイルが新しい重みの隣に残っていると、確率の直し方だけが古いまま
    という「静かに間違った」状態になる。それを防ぐため、校正ファイルに記録した
    重みの指紋と実際の重みを突き合わせ、食い違えば例外にする。
    """
    path = default_path(weights_path)
    if not path.exists():
        if verbose:
            print(
                f"注意: 校正ファイル {path} がありません。確信度は未校正の生の値です"
                "(pos_weightのぶん高めに出ます)。scripts/calibrate.py で作成できます。"
            )
        return Calibration.identity()

    calibration = Calibration.load(path)
    recorded = calibration.source.get("weights_sha256")
    if recorded is None:
        if verbose:
            print(
                f"校正: {path}\n"
                "  警告: この校正ファイルには重みの指紋が記録されていません"
                "(指紋を記録する前に作られたファイル)。今の重みから作られたものか"
                "確認できないため、scripts/calibrate.py で作り直すことを勧めます。"
            )
        return calibration

    actual = file_fingerprint(weights_path)
    if actual != recorded:
        raise StaleCalibrationError(
            f"校正ファイルが、隣にある重みとは別の重みから作られています。\n"
            f"  重み        : {weights_path}(指紋 {actual})\n"
            f"  校正ファイル: {path}(指紋 {recorded} の重み用)\n"
            f"  校正を作った日時: {calibration.source.get('created_at', '不明')}\n"
            f"  校正を作ったラベル: {calibration.source.get('labels_csv', '不明')}\n"
            "この状態で推論すると、確率の直し方だけが古いまま静かに間違った"
            "確信度が出ます。次のどちらかを行ってください:\n"
            "  1) python -m scripts.calibrate で校正を作り直す(推奨)\n"
            f"  2) {path} を削除する(未校正の生の値に戻る)"
        )

    if verbose:
        print(f"校正: {path}(重みの指紋 {actual} と一致)")
    return calibration


def build_source(
    weights_path,
    labels_csv,
    image_size,
    pos_weight,
    pos_weight_source: str,
    pos_weight_cap,
    split: dict,
    fitted_on: str,
    n_fit: int,
) -> dict:
    """校正が「何から作られたか」の記録。古い校正ファイルの取り違えを防ぐ要。

    重みとラベルCSVは中身の指紋(SHA-256の先頭16桁)を取る。どちらかを作り直せば
    指紋が変わるので、あとから食い違いを機械的に検出できる。
    """
    from datetime import datetime, timezone

    source = {
        "weights": str(weights_path),
        "weights_sha256": file_fingerprint(weights_path),
        "image_size": int(image_size),
        "pos_weight": None if pos_weight is None else [float(w) for w in pos_weight],
        # "checkpoint" = 重みに記録されていた値 / "recomputed" = 学習データから計算し直した値
        "pos_weight_source": pos_weight_source,
        "pos_weight_cap": None if pos_weight_cap is None else float(pos_weight_cap),
        "split": dict(split),
        "fitted_on": fitted_on,
        "n_fit": int(n_fit),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if labels_csv is not None:
        source["labels_csv"] = str(labels_csv)
        source["labels_sha256"] = file_fingerprint(labels_csv)
    return source


def load_for_weights_cli(weights_path, verbose: bool = True) -> Calibration:
    """load_for_weights のコマンドライン用。取り違えを、traceback ではなく
    そのまま読めるエラーメッセージにして終了する。"""
    try:
        return load_for_weights(weights_path, verbose=verbose)
    except StaleCalibrationError as e:
        raise SystemExit(f"\nエラー: {e}\n")


def fit(
    logits: np.ndarray,
    targets: np.ndarray,
    pos_weight: np.ndarray = None,
    min_positives: int = MIN_POSITIVES_FOR_FIT,
) -> Calibration:
    """検証データの (ロジット, 正解) から校正を作る。

    pos_weight を渡すと、当てはめの初期値・正則化の中心を -log w にする。
    検証データの陽性が min_positives 未満のラベルは当てはめを行わず、
    -log w の解析的な補正だけを使う。
    """
    logits = np.asarray(logits, dtype="float64")
    targets = np.asarray(targets, dtype="float64")
    per_label = {}
    for i, label in enumerate(LABELS):
        b_prior = 0.0 if pos_weight is None else float(-np.log(max(pos_weight[i], 1e-6)))
        n_pos = int(targets[:, i].sum())

        if n_pos >= min_positives:
            a, b = fit_platt(logits[:, i], targets[:, i], b_init=b_prior)
            method = "platt"
        elif pos_weight is not None:
            a, b = 1.0, b_prior
            method = "prior"
        else:
            a, b = 1.0, 0.0
            method = "none"

        calibrated = sigmoid(a * logits[:, i] + b)
        threshold = find_best_threshold(calibrated, targets[:, i])
        per_label[label] = LabelCalibration(
            a=a, b=b, threshold=threshold, method=method, n_positive=n_pos
        )
    return Calibration(per_label=per_label)


# ---- 校正できているかを測る ----


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """ECE: 「表示した確信度」と「実際に当たった割合」のズレの平均。

    確信度で等間隔のビンに分け、各ビンで |平均確信度 - 正解率| を求め、
    件数で重み付けして平均する。0に近いほど、表示%が実態を表している。
    """
    confidences = np.asarray(confidences, dtype="float64").ravel()
    correct = np.asarray(correct, dtype="float64").ravel()
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        total += mask.sum() / confidences.size * abs(confidences[mask].mean() - correct[mask].mean())
    return float(total)


def reliability_table(confidences: np.ndarray, correct: np.ndarray, bins: int = 10) -> list:
    """信頼度図の中身を表で返す。[(下限, 上限, 件数, 平均確信度, 正解率), ...]"""
    confidences = np.asarray(confidences, dtype="float64").ravel()
    correct = np.asarray(correct, dtype="float64").ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((float(lo), float(hi), 0, float("nan"), float("nan")))
        else:
            rows.append((float(lo), float(hi), n,
                         float(confidences[mask].mean()), float(correct[mask].mean())))
    return rows


def top1_confidence_and_correctness(probs: np.ndarray, targets: np.ndarray):
    """1位ラベルの確信度と、それが実際に正解ラベルに含まれていたかを返す。

    利用者が一番目にする数字(「この日は○○、確信度62%」)が、そのまま
    「その判定が当たっている確率」になっているかを測るために使う。
    """
    probs = np.asarray(probs, dtype="float64")
    targets = np.asarray(targets, dtype="float64")
    top = probs.argmax(axis=1)
    rows = np.arange(len(top))
    return probs[rows, top], targets[rows, top]


def format_reliability(rows: list, indent: str = "  ") -> str:
    """reliability_table の結果を、そのまま印刷できる文字列にする。"""
    lines = [f"{indent}{'確信度の範囲':<16}{'件数':>6}{'平均確信度':>12}{'実際の正解率':>14}{'ズレ':>10}"]
    for lo, hi, n, mean_conf, accuracy in rows:
        if n == 0:
            continue
        gap = mean_conf - accuracy
        flag = "  ←表示が高すぎる" if gap > 0.10 else ""
        lines.append(
            f"{indent}{lo * 100:>3.0f}%〜{hi * 100:>3.0f}%{'':<7}{n:>6}"
            f"{mean_conf * 100:>11.1f}%{accuracy * 100:>13.1f}%{gap * 100:>+9.1f}pt{flag}"
        )
    return "\n".join(lines)


# ---- モデルからロジットを集める(校正を当てはめる側で使う) ----


def collect_logits(model, dataset, rows, device, batch_size: int = 16):
    """指定した行に対して推論し、(生のロジット, 正解ラベル) を numpy で返す。

    sigmoidを通す前のロジットが必要なので、既存の推論経路
    (torch.sigmoid(model(x)))とは別にここで集める。
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    loader = DataLoader(Subset(dataset, rows), batch_size=batch_size)
    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for images, features, labels in loader:
            images = images.to(device)
            outputs = model(images, features.to(device)) if features.numel() else model(images)
            logits_list.append(outputs.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(logits_list), np.concatenate(labels_list)


def summarize(logits: np.ndarray, targets: np.ndarray, calibration: Calibration,
              bins: int = 10) -> dict:
    """校正の前後で、表示%と実際の正解率がどれだけ合うようになったかをまとめる。

    見るのは利用者が実際に目にする数字、つまり「1位ラベルの確信度」。
    """
    raw = sigmoid(np.asarray(logits, dtype="float64"))
    calibrated = calibration.probabilities(logits)

    raw_conf, raw_hit = top1_confidence_and_correctness(raw, targets)
    cal_conf, cal_hit = top1_confidence_and_correctness(calibrated, targets)
    return {
        "n": int(len(targets)),
        "top1_accuracy": float(cal_hit.mean()) if len(cal_hit) else 0.0,
        "raw": {
            "mean_confidence": float(raw_conf.mean()) if len(raw_conf) else 0.0,
            "ece": expected_calibration_error(raw_conf, raw_hit, bins=bins),
            "reliability": reliability_table(raw_conf, raw_hit, bins=bins),
        },
        "calibrated": {
            "mean_confidence": float(cal_conf.mean()) if len(cal_conf) else 0.0,
            "ece": expected_calibration_error(cal_conf, cal_hit, bins=bins),
            "reliability": reliability_table(cal_conf, cal_hit, bins=bins),
        },
    }
