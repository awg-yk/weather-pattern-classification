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

SPLIT_MODES = ("temporal", "by_year", "random")


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


def make_splits(
    df: pd.DataFrame,
    mode: str = "temporal",
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> dict:
    """dfの行番号を train / val / test に振り分けて返す。

    mode:
      temporal : 日付順に前から train → val → test。境界の2か所以外は時間的に離れる
      by_year  : 年ごと丸ごと割り当てる。各分割が全季節を含むので季節の偏りが出ない
      random   : 日付を無視したランダム分割(従来互換。リークするので報告には使わない)
    """
    n = len(df)
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
        # 同時刻の行が混ざっても順序が安定するよう、日付→行番号の順で並べる
        order = list(np.lexsort((df.index.to_numpy(), dates.to_numpy())))

    elif mode == "by_year":
        dates = _require_dates(df)
        years = dates.dt.year.astype(int)
        # 新しい年から順にtest → valへ詰め、残りをtrainにする
        remaining_years = sorted((int(y) for y in years.unique()), reverse=True)
        test_years, val_years = [], []
        filled = 0
        for year in remaining_years:
            if filled < n_test:
                test_years.append(year)
            elif filled < n_test + n_val:
                val_years.append(year)
            else:
                break
            filled += int((years == year).sum())

        train_years = [y for y in remaining_years if y not in test_years and y not in val_years]
        if not train_years:
            raise ValueError(
                f"by_year分割では年が足りません(年: {remaining_years})。"
                "val_ratio/test_ratioを下げるか、--split-mode temporal を使ってください。"
            )

        def rows_of(year_list):
            return [i for i in range(n) if years.iloc[i] in year_list]

        splits = {"train": rows_of(train_years), "val": rows_of(val_years), "test": rows_of(test_years)}
        print(
            f"by_year分割: train={sorted(train_years)} ({len(splits['train'])}件) / "
            f"val={sorted(val_years)} ({len(splits['val'])}件) / "
            f"test={sorted(test_years)} ({len(splits['test'])}件)"
        )
        return splits

    else:
        raise ValueError(f"未知の--split-mode: {mode}(選択肢: {', '.join(SPLIT_MODES)})")

    splits = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }

    if mode == "temporal":
        dates = df["parsed_datetime"]

        def span(name):
            rows = splits[name]
            if not rows:
                return "なし"
            return f"{dates.iloc[rows].min():%Y-%m-%d} 〜 {dates.iloc[rows].max():%Y-%m-%d}"

        print(
            f"temporal分割: train={span('train')} ({len(splits['train'])}件) / "
            f"val={span('val')} ({len(splits['val'])}件) / "
            f"test={span('test')} ({len(splits['test'])}件)"
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
