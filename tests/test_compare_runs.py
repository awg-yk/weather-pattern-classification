"""別のラベルで測った結果どうしを比べてしまう事故を防ぐ仕組みのテスト。

実際にあった事故: 台風ラベルをベストトラックから付け直したあと、付け直す前の
実行(陽性219件)と後の実行(247件)を並べて「+0.192の改善」と読んでしまった。
差はモデルではなくラベルのものだった(docs/2026-08-26-stale-run-comparisons.md)。
"""

from pathlib import Path

import pytest

from scripts.compare_runs import check_comparable, support_signature
from src.labels import LABELS


def make_summary(supports: dict, fingerprint=None) -> dict:
    """foldごとのsupportだけを持つ、最小限のsummary.jsonの中身を作る。"""
    folds = []
    for i in range(3):
        per_label = {
            label: {"f1": 0.5, "support": supports.get(label, [10, 10, 10])[i]}
            for label in LABELS
        }
        folds.append({"test_year": 2023 + i, "per_label": per_label,
                      "macro_f1_all_labels": 0.5, "macro_f1_evaluable": 0.5, "n_eval": 700})
    summary = {"folds": folds}
    if fingerprint:
        summary["labels_fingerprint"] = fingerprint
    return summary


def test_signature_lists_support_per_fold():
    signature = support_signature(make_summary({"typhoon": [98, 62, 59]}))
    assert signature["typhoon"] == [98, 62, 59]


def test_same_labels_are_comparable():
    a = make_summary({"typhoon": [98, 62, 59]})
    b = make_summary({"typhoon": [98, 62, 59]})
    assert check_comparable([(Path("a"), a), (Path("b"), b)]) == []


def test_different_supports_are_flagged():
    """台風を付け直したあとの実行と、付け直す前の実行。"""
    before = make_summary({"typhoon": [98, 62, 59]})
    after = make_summary({"typhoon": [78, 84, 85]})
    problems = check_comparable([(Path("before"), before), (Path("after"), after)])
    assert len(problems) == 1
    path, mismatched, _, _ = problems[0]
    assert path.name == "after"
    assert mismatched == ["typhoon"]


def test_fingerprint_catches_swap_that_keeps_the_count():
    """陽性の枚数を変えずに中身を入れ替えた修正は、supportでは見抜けない。"""
    same_supports = {"typhoon": [98, 62, 59]}
    a = make_summary(same_supports, fingerprint="1111111111111111")
    b = make_summary(same_supports, fingerprint="2222222222222222")
    assert check_comparable([(Path("a"), a), (Path("b"), b)])

    # 指紋が一致していれば通る
    c = make_summary(same_supports, fingerprint="1111111111111111")
    assert check_comparable([(Path("a"), a), (Path("c"), c)]) == []


def test_missing_fingerprint_falls_back_to_support():
    """指紋が無い時期の結果でも、supportが揃っていれば比較を止めない。"""
    a = make_summary({"typhoon": [98, 62, 59]}, fingerprint="1111111111111111")
    b = make_summary({"typhoon": [98, 62, 59]})    # 指紋なし
    assert check_comparable([(Path("a"), a), (Path("b"), b)]) == []


def test_single_run_is_always_comparable():
    assert check_comparable([(Path("a"), make_summary({}))]) == []


def test_every_mismatched_run_is_reported():
    base = make_summary({"okhotsk_high": [68, 62, 58]})
    ok = make_summary({"okhotsk_high": [68, 62, 58]})
    bad = make_summary({"okhotsk_high": [36, 23, 26]})
    problems = check_comparable([(Path("base"), base), (Path("ok"), ok), (Path("bad"), bad)])
    assert [p[0].name for p in problems] == ["bad"]


def test_cross_validate_records_the_labels_fingerprint():
    """cross_validate.py が指紋を書き出すようになっていること。"""
    source = Path("scripts/cross_validate.py").read_text(encoding="utf-8")
    assert "labels_fingerprint" in source
    assert "file_fingerprint" in source


# --- 分割の条件がそろっているか ------------------------------------------

def _summary(val_mode, seed=42, gap_days=3):
    return {
        "config": {"val_mode": val_mode, "seed": seed, "gap_days": gap_days,
                   "split_mode": "loyo", "years": ["2023", "2024", "2025"]},
        "folds": [],
    }


def test_a_different_val_mode_is_reported():
    """分割の条件が違えば知らせること。

    **そろっていないと、モデルの差と学習データ量の差が混ざる。**実測で
    spread は tail より学習データが2割少なかった(906件 対 1157件)。
    片方だけ spread で回した結果を tail の結果と並べると、上積みが
    モデルのものか学習データ量のものか分けられない。
    """
    from pathlib import Path

    from scripts.compare_runs import check_same_split

    problems = check_same_split([
        (Path("cv_baseline"), _summary("spread")),
        (Path("cv_annot"), _summary("tail")),
    ])
    assert len(problems) == 1
    _path, differing = problems[0]
    assert differing["val_mode"] == ("spread", "tail")


def test_the_same_split_raises_nothing():
    from pathlib import Path

    from scripts.compare_runs import check_same_split

    assert check_same_split([
        (Path("a"), _summary("spread")),
        (Path("b"), _summary("spread")),
    ]) == []


def test_seed_and_gap_days_are_checked_too():
    from pathlib import Path

    from scripts.compare_runs import check_same_split

    problems = check_same_split([
        (Path("a"), _summary("tail", seed=42, gap_days=3)),
        (Path("b"), _summary("tail", seed=1, gap_days=7)),
    ])
    assert set(problems[0][1]) == {"seed", "gap_days"}


def test_a_summary_without_config_is_skipped():
    """configを記録する前に作った古い結果でも落ちないこと。"""
    from pathlib import Path

    from scripts.compare_runs import check_same_split

    assert check_same_split([
        (Path("a"), {"folds": []}),
        (Path("b"), _summary("tail")),
    ]) == []
