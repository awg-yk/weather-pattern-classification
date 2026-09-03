"""macro F1 以外の数字も出せることを確かめる。

**macro F1 は「何%当たるか」ではない。**そのまま見せると誤解されるので、
1位正解率・適合率・再現率を並べて出す。数字の意味を取り違えたまま
発表することの方が、指標が1つ足りないことより悪い。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.labels import LABELS


def _summary(path: Path, base: float = 0.669) -> Path:
    """交差検証のまとめを、本物と同じ形で作る。"""
    folds = []
    for offset, year in enumerate((2023, 2024, 2025)):
        per_label = {
            label: {
                "f1": base,
                "precision": min(0.99, base + 0.05),
                "recall": max(0.01, base - 0.05),
                "support": 10 + index,
                "trivial_f1": 0.2,
            }
            for index, label in enumerate(LABELS)
        }
        folds.append({
            "test_year": year,
            "n_eval": 723,
            "top1_accuracy": base + 0.12 + offset * 0.01,
            "macro_f1_evaluable": base + offset * 0.001,
            "macro_f1_all_labels": base,
            "micro_f1": base + 0.09,
            "weighted_f1": base + 0.07,
            "trivial_macro_f1": 0.269,
            "per_label": per_label,
        })
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps({"folds": folds}, ensure_ascii=False), encoding="utf-8")
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "scripts.report_metrics", *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_it_reports_accuracy_alongside_macro_f1(tmp_path):
    """macro F1 だけ出すと『66.9%当たる』と読まれる。"""
    done = _run("--run", str(_summary(tmp_path / "annot")))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1位正解率" in done.stdout, "正解率が出ていない"
    assert "適合率" in done.stdout and "再現率" in done.stdout
    # 数字の意味を、数字のそばに書いておく
    assert "空振り" in done.stdout and "見逃し" in done.stdout


def test_every_label_appears_with_its_support(tmp_path):
    """件数を出さないと、0.3 が『難しいラベル』なのか
    『そもそも数件しかない』のか区別できない。"""
    done = _run("--run", str(_summary(tmp_path / "annot")))
    assert done.returncode == 0, done.stdout + done.stderr
    from src.labels import LABEL_JA
    for label in LABELS:
        assert LABEL_JA[label] in done.stdout, f"{label} が表に無い"


def test_comparison_shows_both_runs_and_the_difference(tmp_path):
    done = _run("--run", str(_summary(tmp_path / "annot", 0.669)),
                "--compare", str(_summary(tmp_path / "baseline", 0.640)),
                "--name", "描き込みあり", "--compare-name", "基準")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "比較" in done.stdout
    assert "+0.029" in done.stdout, f"差が出ていない:\n{done.stdout}"


def test_a_missing_run_says_where_it_looked(tmp_path):
    done = _run("--run", str(tmp_path / "nope"))
    assert done.returncode != 0
    assert "summary.json" in done.stdout + done.stderr


def test_the_csv_and_the_figure_are_written(tmp_path):
    """**パイプに通すと途中で死んでも気づけない。**書き出しまで通す。"""
    run_dir = _summary(tmp_path / "annot")
    csv_path = tmp_path / "out" / "label_metrics.csv"
    png_path = tmp_path / "out" / "pr.png"
    done = _run("--run", str(run_dir), "--csv", str(csv_path), "--plot", str(png_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert csv_path.exists(), "CSVが書かれていない"
    assert png_path.stat().st_size > 1000, "図が書かれていない"

    import pandas as pd
    got = pd.read_csv(csv_path)
    assert len(got) == len(LABELS)
    for column in ("ラベル", "適合率", "再現率", "F1", "件数"):
        assert column in got.columns, f"{column} 列が無い"


@pytest.mark.parametrize("missing", ["top1_accuracy", "trivial_macro_f1"])
def test_it_still_runs_on_older_summaries(tmp_path, missing):
    """古い実行のまとめには入っていない項目がある。落とさずに出せる分だけ出す。"""
    run_dir = _summary(tmp_path / "old")
    path = run_dir / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for fold in data["folds"]:
        fold.pop(missing)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    done = _run("--run", str(run_dir))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "macro F1" in done.stdout


def test_accuracy_comes_with_something_to_compare_it_against(tmp_path):
    """**83%が良いのかどうかは、何もしない場合と比べないと分からない。**
    1枚に複数のラベルが付くので、当たりやすさはラベル数で決まる。"""
    done = _run("--run", str(_summary(tmp_path / "annot")))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "でたらめに1つ選んだ場合" in done.stdout
    assert "一番多いラベルを毎回答えた場合" in done.stdout
    assert "1枚に付く正解ラベルは平均" in done.stdout


def test_the_baselines_match_the_supports(tmp_path):
    """件数から手で計算した値と合うこと。"""
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from scripts.report_metrics import load_summary, top1_baselines

    run_dir = _summary(tmp_path / "annot")
    summary = load_summary(run_dir)
    got = top1_baselines(summary)

    # _summary は support を 10, 11, ... 19 で作る(合計145)、n_eval は723
    total, n = sum(range(10, 10 + len(LABELS))), 723
    assert got["per_chart"][0] == pytest.approx(total / n)
    assert got["chance"][0] == pytest.approx(total / n / len(LABELS))
    assert got["majority"][0] == pytest.approx(max(range(10, 10 + len(LABELS))) / n)


def test_baselines_are_skipped_when_the_count_is_missing(tmp_path):
    """n_eval の無い古いまとめでも落ちない。"""
    run_dir = _summary(tmp_path / "old")
    path = run_dir / "summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for fold in data["folds"]:
        fold.pop("n_eval")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    done = _run("--run", str(run_dir))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1位正解率" in done.stdout
    assert "でたらめに1つ選んだ場合" not in done.stdout
