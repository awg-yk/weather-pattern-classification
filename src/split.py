"""学習用・検証用・テスト用へのデータ分割。

天気図は連続する日どうしが極めて似ているため、日付を無視してランダムに分割すると
「7月3日で学習し、7月4日で評価する」ような状態になり、実質的に見たことのある画像を
当てているだけの水増しされたスコアが出る(時間的リーク)。

そのため既定は時間ブロック分割とし、学習・検証・テストが時間軸上で重ならないようにする。
従来のランダム分割は、過去の実験を再現する用途のために残してある。

    train : モデルのパラメータを学習する
    val   : 早期終了・ベストモデルの選択・判定閾値の探索に使う
    test  : 最終報告の数値を1回だけ測る(上のどれにも使ってはいけない)
"""

import numpy as np
import pandas as pd

SPLIT_MODES = ("temporal", "by_year", "loyo", "random")


def _sizes(n: int, val_ratio: float, test_ratio: float) -> tuple:
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"val_ratio={val_ratio} + test_ratio={test_ratio} が大きすぎて学習データが残りません"
        )
    return n_train, n_val, n_test


def _require_dates(df: pd.DataFrame) -> pd.Series:
    dates = df["parsed_datetime"]
    missing = df.loc[dates.isna(), "filename"]
    if len(missing) > 0:
        raise ValueError(
            f"日付を取り出せない行が{len(missing)}件あります(時間ブロック分割には日付が必須です)。\n"
            f"  例: {list(missing[:5])}\n"
            "ファイル名にYYYYMMDDHHが含まれているか確認してください。"
            "日付なしで分割したい場合は --split-mode random を指定できますが、"
            "時間的リークが起きるため報告用の数値には使えません。"
        )
    return dates


def _split_days_by_count(dates: pd.Series, rows: list, targets: list) -> list:
    """rowsを日単位でまとめ、日付順にtargets件ずつのブロックに切り分ける。

    同じ日の00Z/12Zはほぼ同一の画像なので、必ず同じブロックに入れる。
    累積件数で境界を決めるため、後ろの日が前のブロックに戻ることはない。
    """
    rows_by_day = {}
    for i in rows:
        rows_by_day.setdefault(dates.iloc[i].normalize(), []).append(i)

    ordered_days = sorted(rows_by_day)
    cumulative = np.cumsum([len(rows_by_day[d]) for d in ordered_days])

    blocks, start = [], 0
    boundary = 0
    for target in targets:
        boundary += target
        end = int(np.searchsorted(cumulative, boundary, side="left")) + 1
        end = min(max(end, start), len(ordered_days))
        blocks.append([i for d in ordered_days[start:end] for i in rows_by_day[d]])
        start = end
    blocks.append([i for d in ordered_days[start:] for i in rows_by_day[d]])
    return blocks


def make_splits(
    df: pd.DataFrame,
    mode: str = "temporal",
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
    test_year=None,
    gap_days: int = 3,
) -> dict:
    """dfの行番号を train / val / test に振り分けて返す。

    mode:
      temporal : 日付順に前から train → val → test。境界の2か所以外は時間的に離れる
      by_year  : 年ごと丸ごと割り当てる。各分割が全季節を含むので季節の偏りが出ない
      loyo     : test_yearの1年をテストに、残りの年を日付順にtrain/valへ分ける。
                 テストが必ず通年になるので、季節で偏って評価できないラベルが出ない
      random   : 日付を無視したランダム分割(従来互換。リークするので報告には使わない)

    gap_days: loyoで、テスト年からこの日数以内にある学習データを除外する。
              年をまたぐ境界(12月末と1月初)は数日差で天気図がほぼ同じになるため。
    """
    n = len(df)
    if mode == "loyo":
        if test_year is None:
            raise ValueError("--split-mode loyo では --test-year の指定が必要です")
        dates = _require_dates(df)
        years = dates.dt.year.astype(int)
        test_rows = [i for i in range(n) if years.iloc[i] == int(test_year)]
        if not test_rows:
            raise ValueError(
                f"{test_year}年の画像がありません(対象の年: {sorted(set(years))})"
            )

        pool = [i for i in range(n) if years.iloc[i] != int(test_year)]

        # テスト年の前後(前年12月末・翌年1月初)は、テスト年の端の日と数日しか
        # 離れておらず天気図がほとんど同じになる。学習側からその期間を取り除く。
        if gap_days > 0:
            distance = min_days_to_other_split(df, pool, test_rows)
            purged = [i for i, d in zip(pool, distance) if d <= gap_days]
            pool = [i for i, d in zip(pool, distance) if d > gap_days]
            if purged:
                print(f"  テスト年から{gap_days}日以内の学習データ{len(purged)}件を除外しました")

        # 残りの年を日付順に並べ、後ろ val_ratio ぶんを検証用にする
        n_val_pool = int(len(pool) * val_ratio)
        train_rows, val_rows = _split_days_by_count(
            dates, pool, [len(pool) - n_val_pool]
        )
        splits = {"train": train_rows, "val": val_rows, "test": test_rows}

        dates_all = df["parsed_datetime"]
        print(
            f"loyo分割(テスト={test_year}年): "
            f"train({len(train_rows)}件, {dates_all.iloc[train_rows].min():%Y-%m-%d}"
            f"〜{dates_all.iloc[train_rows].max():%Y-%m-%d}) / "
            f"val({len(val_rows)}件, {dates_all.iloc[val_rows].min():%Y-%m-%d}"
            f"〜{dates_all.iloc[val_rows].max():%Y-%m-%d}) / "
            f"test({len(test_rows)}件, {test_year}年通年)"
        )
        return splits

    n_train, n_val, n_test = _sizes(n, val_ratio, test_ratio)

    if mode == "random":
        # torch.random_splitと同じ並び替えを使うので、test_ratio=0なら従来と同一の分割になる。
        # torchはこの分岐でしか要らないので、診断だけしたいときに重い依存を持ち込まないよう
        # ここでインポートする。
        import torch

        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(n, generator=generator).tolist()

    elif mode == "temporal":
        dates = _require_dates(df)
        # 日単位でまとめてから割り当てる。JMAアーカイブは1日2回(00Z/12Z)あり、
        # 同じ日の2枚はほぼ同一なので、境界で引き裂かれて別々の分割に入るのを防ぐ。
        train_rows, val_rows, test_rows = _split_days_by_count(
            dates, list(range(n)), [n_train, n_val]
        )
        splits = {"train": train_rows, "val": val_rows, "test": test_rows}

    elif mode == "by_year":
        dates = _require_dates(df)
        years = dates.dt.year.astype(int)
        all_years = sorted(int(y) for y in years.unique())

        # 年数そのものを比率で配分する。件数で貪欲に詰めると、1年の件数が目標を
        # 大きく超えたときに丸ごと入ってしまい、後続の分割が空になる。
        n_years = len(all_years)
        n_test_y = max(1, round(n_years * test_ratio)) if test_ratio > 0 else 0
        n_val_y = max(1, round(n_years * val_ratio)) if val_ratio > 0 else 0
        if n_years - n_test_y - n_val_y < 1:
            raise ValueError(
                f"by_year分割には年が足りません(対象の年: {all_years})。"
                "train/val/testそれぞれに最低1年ずつ必要です。"
                "--split-mode temporal を使うか、val_ratio/test_ratioを下げてください。"
            )

        # 古い年から train → val → test(将来を予測する使い方に合わせて時系列順)
        train_years = all_years[: n_years - n_val_y - n_test_y]
        val_years = all_years[n_years - n_val_y - n_test_y : n_years - n_test_y]
        test_years = all_years[n_years - n_test_y :] if n_test_y else []

        def rows_of(year_list):
            return [i for i in range(n) if years.iloc[i] in year_list]

        splits = {"train": rows_of(train_years), "val": rows_of(val_years), "test": rows_of(test_years)}

    else:
        raise ValueError(f"未知の--split-mode: {mode}(選択肢: {', '.join(SPLIT_MODES)})")

    if mode == "random":
        splits = {
            "train": order[:n_train],
            "val": order[n_train : n_train + n_val],
            "test": order[n_train + n_val :],
        }

    # 比率を指定したのに空になった分割があれば、黙って進めず知らせる
    # (空のvalで学習するとベストモデルを選べない)
    for name, ratio in (("val", val_ratio), ("test", test_ratio)):
        if ratio > 0 and not splits[name]:
            raise ValueError(
                f"{name}が0件になりました(--split-mode {mode}, {name}_ratio={ratio})。"
                "--split-mode temporal を使うか、比率を調整してください。"
            )

    dates = df["parsed_datetime"]

    def describe(name):
        rows = splits[name]
        if not rows:
            return "なし"
        label = f"{len(rows)}件"
        if mode != "random" and dates.notna().all():
            label += f", {dates.iloc[rows].min():%Y-%m-%d}〜{dates.iloc[rows].max():%Y-%m-%d}"
        return label

    print(
        f"{mode}分割: train({describe('train')}) / "
        f"val({describe('val')}) / test({describe('test')})"
    )
    return splits


def min_days_to_other_split(df: pd.DataFrame, rows_a, rows_b) -> pd.Series:
    """rows_aの各行について、rows_bの中で最も日付が近いものとの日数差を返す。

    時間的リークの度合いを測るための診断用。0や1が並ぶなら、評価対象と
    ほぼ同じ日の画像で学習してしまっている。
    """
    if not rows_a or not rows_b:
        return pd.Series(dtype=float)

    a = df["parsed_datetime"].iloc[rows_a]
    b = np.sort(df["parsed_datetime"].iloc[rows_b].to_numpy())

    # 各aについて、ソート済みbの中で最も近い値との差(日数)
    idx = np.searchsorted(b, a.to_numpy())
    diffs = []
    for value, position in zip(a.to_numpy(), idx):
        candidates = []
        if position > 0:
            candidates.append(abs(value - b[position - 1]))
        if position < len(b):
            candidates.append(abs(value - b[position]))
        diffs.append(min(candidates) / np.timedelta64(1, "D"))
    return pd.Series(diffs, index=a.index)
