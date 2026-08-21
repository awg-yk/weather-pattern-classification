"""データ分割のテスト。

ここが壊れると、学習した画像で評価してしまい、実力より高いスコアが出る。
しかも学習も評価も正常に完了するので、数字を見ても気づけない。
実際に最初の0.75はそれで出た値だった。止まるバグより危険なので、
分割については「重ならないこと」を機械的に確かめておく。
"""

import pandas as pd
import pytest

from src.split import make_splits, min_days_to_other_split

YEARS = (2023, 2024, 2025)


@pytest.fixture
def frame():
    """1日2回(00Z/12Z)、3年分の観測日時を持つDataFrame。"""
    stamps = pd.date_range("2023-01-01", "2025-12-31 12:00", freq="12h")
    return pd.DataFrame(
        {"filename": [f"Js_{t:%Y%m%d%H}.png" for t in stamps], "parsed_datetime": stamps}
    )


def _assert_disjoint(splits):
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert not train & val, "学習と検証が重複している"
    assert not train & test, "学習とテストが重複している"
    assert not val & test, "検証とテストが重複している"


@pytest.mark.parametrize("val_mode", ["tail", "spread"])
@pytest.mark.parametrize("test_year", YEARS)
def test_loyo_splits_are_disjoint(frame, val_mode, test_year):
    splits = make_splits(frame, mode="loyo", test_year=test_year, val_mode=val_mode)
    _assert_disjoint(splits)


@pytest.mark.parametrize("val_mode", ["tail", "spread"])
def test_loyo_test_set_is_exactly_the_held_out_year(frame, val_mode):
    splits = make_splits(frame, mode="loyo", test_year=2024, val_mode=val_mode)
    years = frame["parsed_datetime"].dt.year
    assert set(years.iloc[splits["test"]]) == {2024}
    assert 2024 not in set(years.iloc[splits["train"]])
    assert 2024 not in set(years.iloc[splits["val"]])


@pytest.mark.parametrize("val_mode", ["tail", "spread"])
def test_training_data_keeps_its_distance_from_the_test_year(frame, val_mode):
    gap = 3
    splits = make_splits(
        frame, mode="loyo", test_year=2024, gap_days=gap, val_mode=val_mode
    )
    distance = min_days_to_other_split(frame, splits["train"], splits["test"])
    assert distance.min() > gap, "テスト年の直前直後の日が学習に残っている"


def test_spread_validation_covers_every_month(frame):
    """tailは学習期間の末尾しか取らないので、必ず秋冬に偏る。

    夏に集中するラベル(オホーツク海高気圧・太平洋高気圧型)は検証データに
    ほとんど現れず、閾値もモデル選択もその季節を見ないまま決まってしまう。
    """
    spread = make_splits(frame, mode="loyo", test_year=2023, val_mode="spread")
    months = set(frame["parsed_datetime"].iloc[spread["val"]].dt.month)
    assert months == set(range(1, 13)), f"検証データに欠けている月がある: {sorted(months)}"

    tail = make_splits(frame, mode="loyo", test_year=2023, val_mode="tail")
    tail_months = set(frame["parsed_datetime"].iloc[tail["val"]].dt.month)
    assert len(tail_months) < 12, "tailが通年になっている(前提が変わった)"


def test_spread_validation_is_separated_from_training(frame):
    """抜いた週の前後を学習から除いていないと、隣接日で実質的に学習してしまう。"""
    gap = 3
    splits = make_splits(
        frame, mode="loyo", test_year=2023, gap_days=gap, val_mode="spread"
    )
    distance = min_days_to_other_split(frame, splits["train"], splits["val"])
    assert distance.min() > gap


def test_same_day_charts_stay_together(frame):
    """同じ日の00Zと12Zはほぼ同じ絵なので、別々の分割に入ってはいけない。"""
    splits = make_splits(frame, mode="temporal", val_ratio=0.2, test_ratio=0.2)
    days = frame["parsed_datetime"].dt.normalize()
    seen = {}
    for name in ("train", "val", "test"):
        for row in splits[name]:
            day = days.iloc[row]
            assert seen.setdefault(day, name) == name, f"{day:%Y-%m-%d} が分割をまたいでいる"


def test_every_row_is_used_exactly_once(frame):
    splits = make_splits(frame, mode="temporal", val_ratio=0.2, test_ratio=0.2)
    assigned = [row for name in ("train", "val", "test") for row in splits[name]]
    assert sorted(assigned) == list(range(len(frame)))


def test_by_year_gives_each_split_whole_years(frame):
    splits = make_splits(frame, mode="by_year", val_ratio=0.34, test_ratio=0.33)
    years = frame["parsed_datetime"].dt.year
    groups = [set(years.iloc[splits[name]]) for name in ("train", "val", "test")]
    assert all(groups), "空の分割がある"
    assert not groups[0] & groups[1] and not groups[1] & groups[2]


def test_loyo_rejects_a_year_with_no_data(frame):
    with pytest.raises(ValueError, match="2019"):
        make_splits(frame, mode="loyo", test_year=2019)


def test_missing_dates_are_reported_not_ignored():
    """日付が取れない行を黙って捨てると、分割が意図と違うものになる。"""
    df = pd.DataFrame({"filename": ["broken.png"], "parsed_datetime": [pd.NaT]})
    with pytest.raises(ValueError, match="日付"):
        make_splits(df, mode="temporal")


def test_trivial_macro_f1_matches_the_all_positive_baseline():
    """macro F1には「全部を陽性と答える」だけで得られる下駄がある。

    これを引かずに絶対値だけを見ていたため、自明な予測を下回った実行(macro F1
    0.250 に対して基準 0.29)を「低いが学習はできている」と読み違えていた。
    """
    import numpy as np

    from src.evaluate import trivial_macro_f1

    labels = np.zeros((1000, 2))
    labels[:420, 0] = 1  # 出現率0.42 → 2p/(1+p) = 0.592
    labels[:100, 1] = 1  # 出現率0.10 → 0.182
    macro, per_label = trivial_macro_f1(labels)
    assert per_label[0] == pytest.approx(2 * 0.42 / 1.42)
    assert per_label[1] == pytest.approx(2 * 0.10 / 1.10)
    assert macro == pytest.approx((per_label[0] + per_label[1]) / 2)


def test_trivial_macro_f1_skips_labels_that_never_occur():
    """1件も出現しないラベルは基準の平均から外す(F1を測れないため)。"""
    import numpy as np

    from src.evaluate import trivial_macro_f1

    labels = np.zeros((100, 2))
    labels[:20, 0] = 1
    macro, _ = trivial_macro_f1(labels)
    assert macro == pytest.approx(2 * 0.2 / 1.2)


def test_paired_diff_flags_when_folds_disagree():
    """foldごとに対応させた差を見る。符号が揃うかどうかが判断材料になる。

    平均の差が標準偏差より小さくても、全foldで同じ向きなら実質的な改善と読める。
    逆に符号がばらつくなら、平均が動いていてもたまたまの可能性が高い。
    """
    from scripts.compare_runs import paired_diff

    consistent = paired_diff([0.40, 0.41, 0.44], [0.45, 0.42, 0.45], [2023, 2024, 2025])
    assert "全foldで同符号" in consistent
    assert "+0.023" in consistent

    noisy = paired_diff([0.40, 0.45, 0.42], [0.44, 0.41, 0.46], [2023, 2024, 2025])
    assert "ばらつく" in noisy


def test_summary_is_readable_whichever_encoding_it_was_written_in(tmp_path):
    """Windowsで作られた過去の結果(cp932)も読めること。

    書き出し側でencodingを指定していなかった時期があり、その結果は環境の既定で
    保存されている。ラベル名が日本語なので、UTF-8決め打ちで読むと落ちる。
    比較のたびに学習をやり直すわけにはいかない。
    """
    import json

    from scripts.compare_runs import load

    payload = {"folds": [], "note": "西高東低（冬型）"}
    for encoding in ("cp932", "utf-8"):
        path = tmp_path / f"{encoding}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding=encoding)
        assert load(path)["note"] == "西高東低（冬型）"


def test_ensemble_refuses_to_blend_rows_that_do_not_line_up():
    """天気図と格子で行の並びが揃っていなければ止めること。

    揃っていない状態で混ぜると、ある日の天気図の予測を別の日の気圧場の予測と
    足すことになる。エラーにはならず、静かに無意味な数字が出るので、
    ここで気づけないと最後まで気づけない。
    """
    from scripts.ensemble_chart_grid import require_aligned

    stamps = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    aligned = pd.DataFrame({"parsed_datetime": stamps})
    require_aligned(aligned, pd.DataFrame({"parsed_datetime": stamps}))

    with pytest.raises(SystemExit, match="行数が違います"):
        require_aligned(aligned, pd.DataFrame({"parsed_datetime": stamps[:2]}))

    with pytest.raises(SystemExit, match="並びが揃っていません"):
        require_aligned(aligned, pd.DataFrame({"parsed_datetime": stamps[::-1]}))


def test_blend_weight_is_chosen_per_label():
    """得意分野が逆のラベルを、ひとつの重みで妥協させないこと。

    一律の重みだと、天気図が強い台風(0.764対0.492)と格子が強いオホーツク海高気圧
    (0.131対0.345)が同じ値を共有し、両方が損をした -- 混合0.607は天気図単独0.619を
    下回った。ラベルごとに選べば、弱いほうのモデルは検証の時点で自然に外れる。
    """
    import numpy as np

    from scripts.ensemble_chart_grid import per_label_weights

    targets = np.zeros((200, 2))
    targets[:80, 0] = 1
    targets[:60, 1] = 1
    # 1列目は天気図が当て格子が外す、2列目はその逆。外すほうは単に無情報なのでは
    # なく正解の逆を出す -- 無情報(定数)だと、ごく小さい重みでも分離できてしまい
    # 最適な重みが一意に決まらない。
    chart = np.stack([targets[:, 0], 1 - targets[:, 1]], axis=1)
    grid = np.stack([1 - targets[:, 0], targets[:, 1]], axis=1)

    weights, thresholds = per_label_weights(
        chart, grid, targets, np.round(np.arange(0, 1.01, 0.05), 2)
    )
    assert weights[0] < 0.5, "天気図が当てているラベルで格子に寄っている"
    assert weights[1] > 0.5, "格子が当てているラベルで天気図に寄っている"
    assert (thresholds > 0).all()


def test_blend_weight_stays_neutral_for_labels_with_no_positives():
    """検証データに1件も出ないラベルは、選びようがないので混ぜない。"""
    import numpy as np

    from scripts.ensemble_chart_grid import per_label_weights

    targets = np.zeros((50, 1))
    weights, _ = per_label_weights(
        np.full((50, 1), 0.5), np.full((50, 1), 0.5), targets, np.array([0.0, 0.5, 1.0])
    )
    assert weights[0] == 0.0


def test_preflight_reports_labels_absent_from_validation(tmp_path, capsys):
    """検証データに1件も出ないラベルを、学習を始める前に知らせること。

    そのラベルについては閾値もモデル選択も混合の重みも決めようがなく、決めれば
    でたらめになる。実際 okhotsk_high は初夏の型で、既定の--val-mode tail では
    検証(学習期間の末尾=秋冬)に現れないまま、通年のテストで評価されていた。
    1foldに数十分かかるので、走らせてから気づくのでは遅い。
    """
    import subprocess
    import sys

    rows = []
    for date in pd.date_range("2023-01-01", "2025-12-31", freq="D"):
        labels = ["migratory_high"]
        if date.month == 6 and date.day % 5 == 0:
            labels.append("okhotsk_high")  # 初夏にしか出ない
        rows.append({"filename": f"Js_{date.strftime('%Y%m%d')}00.png", "label": "|".join(labels)})
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    def run(*extra):
        return subprocess.run(
            [sys.executable, "-m", "scripts.preflight", "--labels", str(labels_csv),
             "--input-mode", "era5-grid", "--era5-grid-dir", str(tmp_path / "missing"),
             "--years", "2023", "2024", "2025", *extra],
            capture_output=True, text=True,
        ).stdout

    assert "オホーツク海高気圧" in run().split("警告")[1]
    # 通年から抜き取れば、そのラベルも検証に入る
    assert "警告" not in run("--val-mode", "spread")


def test_loyo_says_which_years_are_missing_instead_of_failing_on_a_date(tmp_path):
    """テスト年しか渡していないとき、理由の分かる形で止めること。

    学習に回す年が無いと分割が空になり、その先の日付整形が
    「NaTType does not support strftime」で落ちる。何が悪いのか読み取れない。
    """
    dates = pd.date_range("2026-01-01", "2026-04-30", freq="D")
    df = pd.DataFrame({
        "filename": [f"Js_{d.strftime('%Y%m%d')}00.png" for d in dates],
        "parsed_datetime": dates,
    })
    with pytest.raises(ValueError, match="学習に回す年"):
        make_splits(df, mode="loyo", val_ratio=0.2, test_ratio=0.0, test_year=2026)


def test_label_stats_counts_the_pair_that_should_imply_a_label(tmp_path, capsys):
    """「この2つが揃うなら本来こう付くはず」を数えられること。

    件数だけを見ても、そのラベルが難しいのか基準が揺れているのかは分からない。
    futatsudama_low は106件あるが japan_sea_low と nankigan_low の両方が付いた
    ものは0件で、個別の2ラベルを置き換える運用になっていた -- こういう規約は
    気象の知識で決まるので、引数で指定する。
    """
    import subprocess
    import sys

    rows = (["futatsudama_low"] * 5
            + ["japan_sea_low"] * 4
            + ["japan_sea_low|nankigan_low"] * 3)
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame({
        "filename": [f"Js_2023{i // 28 + 1:02d}{i % 28 + 1:02d}00.png" for i in range(len(rows))],
        "label": rows,
    }).to_csv(labels_csv, index=False)

    out = subprocess.run(
        [sys.executable, "-m", "scripts.label_stats", "--labels", str(labels_csv),
         "--label", "futatsudama_low",
         "--implied-by", "japan_sea_low", "nankigan_low", "--list-exceptions"],
        capture_output=True, text=True,
    ).stdout
    assert "あり: 0件" in out
    assert "なし: 3件" in out
    # 違反は「組み合わせが揃っていて対象が付いていない」行。多数派か少数派かは関係ない
    assert out.count("japan_sea_low|nankigan_low") == 3
