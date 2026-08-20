"""
Colab上で画像を1枚ずつ表示し、チェックボックスで複数ラベルを選んで「決定」ボタンで
記録する簡易ツール(マルチラベル対応)。

1枚の天気図に複数のパターンが同時に当てはまることがあるため(例: 西高東低かつ
日本海低気圧)、複数選択できるようにしている。ラベルは data/labels.csv の
label列にパイプ区切り("winter_pressure_pattern|japan_sea_low")で保存される。

Colabのノートブックセルで以下のように使う:

    import sys
    sys.path.append("/content/weather-pattern-classification")
    from scripts.label_tool import run_labeling_session, run_review_session

    # 1) 未ラベルの新しい画像にラベルを付ける
    run_labeling_session(
        images_dir="data/processed/jma",
        labels_csv="data/labels.csv",
    )

    # 2) 既にラベル済みの画像を見直して、追加のタグを付け足す
    #    (例: futatsudama_lowが少ないので、japan_sea_low/nankigan_lowが
    #     付いている画像を見直して該当すれば追加する)
    run_review_session(
        images_dir="data/processed/jma",
        labels_csv="data/labels.csv",
        filter_labels=["japan_sea_low", "nankigan_low"],
    )

    # 3) 新しいラベルを追加した後、既存の画像を見直して付け足す
    #    (例: okhotsk_highを追加した場合、出現しやすい5〜8月の画像だけに
    #     絞って見直す)
    run_review_session(
        images_dir="data/processed/jma",
        labels_csv="data/labels.csv",
        month_range=(5, 8),
    )

- 既にラベル済みのファイルは新規セッション(1)では自動的にスキップされる
- 「決定」ボタンで選択中のチェックボックスの内容を保存して次の画像に進む
- 「わからない/該当なし」ボタンで unclassified として記録する
- 「戻る」ボタンで直前の1件を取り消せる(新規セッションのみ)
- images_dirはPNG(data/processed/jma等)・JPEG(NDL手動アーカイブ由来)の
  どちらも拾う
"""

import re
from pathlib import Path

import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as widgets
from PIL import Image

from src.labels import LABELS, LABEL_JA

SKIP_LABEL = "unclassified"
SKIP_LABEL_JA = "わからない/該当なし"

# 対応する画像拡張子。data/processed/jma配下はPNGだが、NDL手動アーカイブ
# (data/raw/ndl_manual/jpg)由来の画像はJPEGなので両方拾えるようにする。
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")

# ファイル名からYYYYMMDDHH形式の日付を抜き出す。
# JMA取得分("Js_2025010100.png")・NDL手動アーカイブ分
# ("Js_2001070300_page001.jpg")のどちらの命名にも対応する。
_DATE_PATTERN = re.compile(r"(\d{10})")


def _list_images(images_dir: Path) -> list:
    files = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(pattern))
    return sorted(files)


def _guess_date(filename: str) -> str:
    match = _DATE_PATTERN.search(filename)
    return match.group(1) if match else ""


def _load_labeled_filenames(labels_csv: Path) -> set:
    if not labels_csv.exists():
        return set()
    df = pd.read_csv(labels_csv)
    return set(df["filename"])


def _append_labels(labels_csv: Path, filename: str, labels: list, source: str) -> None:
    file_exists = labels_csv.exists()
    date_guess = _guess_date(filename)
    label_field = "|".join(labels) if labels else SKIP_LABEL
    row_df = pd.DataFrame([{
        "filename": filename, "label": label_field, "source": source, "date": date_guess,
    }])
    row_df.to_csv(labels_csv, mode="a", header=not file_exists, index=False)


def _remove_last_row(labels_csv: Path) -> str | None:
    if not labels_csv.exists():
        return None
    df = pd.read_csv(labels_csv)
    if len(df) == 0:
        return None
    removed_filename = df.iloc[-1]["filename"]
    df.iloc[:-1].to_csv(labels_csv, index=False)
    return removed_filename


def _make_checkbox_grid():
    checkboxes = {
        label: widgets.Checkbox(value=False, description=LABEL_JA[label], indent=False)
        for label in LABELS
    }
    grid = widgets.GridBox(
        list(checkboxes.values()),
        layout=widgets.Layout(grid_template_columns="repeat(2, 260px)"),
    )
    return checkboxes, grid


def run_labeling_session(images_dir: str, labels_csv: str, source: str = "jma_manual"):
    """未ラベルの画像に、複数選択でラベルを付けていく。"""
    images_dir = Path(images_dir)
    labels_csv = Path(labels_csv)

    all_files = _list_images(images_dir)
    labeled = _load_labeled_filenames(labels_csv)
    remaining = [f for f in all_files if f.name not in labeled]

    state = {"idx": 0}
    output = widgets.Output()
    progress_label = widgets.Label()
    checkboxes, grid = _make_checkbox_grid()

    def update_progress():
        done = len(all_files) - len(remaining) + state["idx"]
        progress_label.value = f"{done} / {len(all_files)} 完了 (残り {len(remaining) - state['idx']} 枚)"

    def reset_checkboxes():
        for cb in checkboxes.values():
            cb.value = False

    def show_current():
        reset_checkboxes()
        with output:
            clear_output(wait=True)
            if state["idx"] >= len(remaining):
                print("すべてラベル付けが完了しました。")
                return
            img_path = remaining[state["idx"]]
            print(img_path.name)
            display(Image.open(img_path).resize((500, 375)))
        update_progress()

    def advance(selected_labels):
        if state["idx"] >= len(remaining):
            return
        img_path = remaining[state["idx"]]
        _append_labels(labels_csv, img_path.name, selected_labels, source)
        state["idx"] += 1
        show_current()

    def on_confirm_click(_):
        selected = [label for label, cb in checkboxes.items() if cb.value]
        if not selected:
            return  # 何も選ばれていない場合は「わからない」ボタンを使ってもらう
        advance(selected)

    def on_skip_click(_):
        advance([])

    def on_back_click(_):
        if state["idx"] == 0:
            return
        removed = _remove_last_row(labels_csv)
        if removed:
            state["idx"] -= 1
        show_current()

    confirm_button = widgets.Button(description="決定", button_style="success")
    confirm_button.on_click(on_confirm_click)
    skip_button = widgets.Button(description=SKIP_LABEL_JA, button_style="warning")
    skip_button.on_click(on_skip_click)
    back_button = widgets.Button(description="戻る", button_style="danger")
    back_button.on_click(on_back_click)

    controls = widgets.HBox([confirm_button, skip_button, back_button])

    display(progress_label, output, grid, controls)
    show_current()


def run_review_session(
    images_dir: str,
    labels_csv: str,
    filter_labels: list | None = None,
    month_range: tuple | None = None,
):
    """既にラベル済みの画像を見直し、チェックボックスの状態を編集して上書き保存する。

    filter_labels を指定すると、そのいずれかのラベルが付いている画像だけに絞り込む
    (例: 少数派のfutatsudama_lowを見直したい場合、関連しそうなjapan_sea_low/
    nankigan_lowが付いている画像だけを対象にする、など)。

    month_range を (開始月, 終了月) で指定すると、date列(YYYYMMDDHH)の月がその
    範囲に入る画像だけに絞り込む(例: オホーツク海高気圧の見直しなら
    month_range=(5, 8) で梅雨〜盛夏の画像だけに絞れる)。
    """
    images_dir = Path(images_dir)
    labels_csv = Path(labels_csv)

    if not labels_csv.exists():
        print("labels.csv がまだありません。")
        return

    df = pd.read_csv(labels_csv)
    df["parsed"] = df["label"].apply(lambda s: str(s).split("|"))

    mask = pd.Series(True, index=df.index)
    if filter_labels:
        mask &= df["parsed"].apply(lambda ls: any(l in filter_labels for l in ls))
    if month_range:
        start_month, end_month = month_range
        month = df["date"].astype(str).str.slice(4, 6)
        month_num = pd.to_numeric(month, errors="coerce")
        mask &= month_num.between(start_month, end_month)
    target_indices = df[mask].index.tolist()

    state = {"pos": 0}
    output = widgets.Output()
    progress_label = widgets.Label()
    checkboxes, grid = _make_checkbox_grid()

    def update_progress():
        progress_label.value = f"見直し中: {state['pos'] + 1} / {len(target_indices)}"

    def set_checkboxes_from_row(row_idx):
        current_labels = set(df.at[row_idx, "parsed"])
        for label, cb in checkboxes.items():
            cb.value = label in current_labels

    def show_current():
        with output:
            clear_output(wait=True)
            if state["pos"] >= len(target_indices):
                print("見直しが完了しました。")
                return
            row_idx = target_indices[state["pos"]]
            filename = df.at[row_idx, "filename"]
            img_path = images_dir / filename
            print(f"{filename}  現在のラベル: {df.at[row_idx, 'label']}")
            if img_path.exists():
                display(Image.open(img_path).resize((500, 375)))
            else:
                print("(画像ファイルが見つかりません)")
        set_checkboxes_from_row(target_indices[state["pos"]]) if state["pos"] < len(target_indices) else None
        update_progress()

    def save_and_advance(_):
        if state["pos"] >= len(target_indices):
            return
        row_idx = target_indices[state["pos"]]
        selected = [label for label, cb in checkboxes.items() if cb.value]
        df.at[row_idx, "label"] = "|".join(selected) if selected else SKIP_LABEL
        df.drop(columns=["parsed"]).to_csv(labels_csv, index=False)
        df["parsed"] = df["label"].apply(lambda s: str(s).split("|"))
        state["pos"] += 1
        show_current()

    def skip_without_saving(_):
        state["pos"] += 1
        show_current()

    def go_back(_):
        if state["pos"] == 0:
            return
        state["pos"] -= 1
        show_current()

    save_button = widgets.Button(description="保存して次へ", button_style="success")
    save_button.on_click(save_and_advance)
    next_button = widgets.Button(description="変更せず次へ")
    next_button.on_click(skip_without_saving)
    back_button = widgets.Button(description="戻る", button_style="danger")
    back_button.on_click(go_back)

    controls = widgets.HBox([save_button, next_button, back_button])

    display(progress_label, output, grid, controls)
    show_current()


def run_binary_review_session(
    images_dir: str,
    labels_csv: str,
    out_csv: str,
    label: str = "okhotsk_high",
    sample: int = 200,
    seed: int = 42,
    years: list | None = None,
):
    """1つのラベルについて「あり/なし」だけを、元の答えを伏せて判定し直す。

    元のラベルを表示しないのが要点。自分が以前どう答えたかが見えていると、
    無意識にそれへ引きずられ(アンカリング)、一致率が実際より高く出てしまう。

    伏せたまま2回目を付ければ、1回目との一致率が測れる。これは人間でも判断が
    揺れる度合いであり、モデルの性能がどこまで伸びうるかの上限にあたる。
    一致率が90%なら、それを超えるF1は原理的に期待できない。

    結果は labels.csv とは別のファイルに書く。突き合わせて確認するまで、
    元のラベルには一切手を触れない。

    途中で止めても、既に答えた分は飛ばして続きから再開できる。
    """
    images_dir = Path(images_dir)
    out_path = Path(out_csv)

    df = pd.read_csv(labels_csv)
    df["parsed"] = df["label"].apply(lambda s: str(s).split("|"))
    if years:
        years = {int(y) for y in years}
        year_of = df["filename"].str.extract(_DATE_PATTERN)[0].str.slice(0, 4)
        df = df[pd.to_numeric(year_of, errors="coerce").isin(years)]

    # 画像が実在する行だけを対象にする
    df = df[df["filename"].apply(lambda f: (images_dir / f).exists())].reset_index(drop=True)
    if df.empty:
        print(f"{images_dir} に対象の画像がありません。")
        return

    # 無作為抽出。seedを固定しているので、中断して開き直しても同じ200枚になる。
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)

    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path)["filename"])
    remaining = [f for f in df["filename"] if f not in done]

    if not remaining:
        print(f"{len(done)}枚すべて判定済みです。scripts/compare_review.py で突き合わせてください。")
        return

    state = {"idx": 0}
    output = widgets.Output()
    progress_label = widgets.Label()

    def update_progress():
        finished = len(done) + state["idx"]
        progress_label.value = f"{finished} / {len(df)} 枚  ({LABEL_JA[label]}の判定)"

    def show_current():
        with output:
            clear_output(wait=True)
            if state["idx"] >= len(remaining):
                print(f"完了しました。{out_path} に書き出しています。")
                return
            filename = remaining[state["idx"]]
            # 日付は表示する(季節が分からないと判断できないため)が、
            # 以前付けた答えは表示しない
            match = _DATE_PATTERN.search(filename)
            stamp = match.group(1) if match else filename
            print(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}Z")
            display(Image.open(images_dir / filename).resize((720, 540)))
        update_progress()

    def record(answer):
        if state["idx"] >= len(remaining):
            return
        filename = remaining[state["idx"]]
        row = pd.DataFrame([{"filename": filename, "answer": answer}])
        row.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        state["idx"] += 1
        show_current()

    yes_button = widgets.Button(description=f"あり (1)", button_style="success")
    yes_button.on_click(lambda _: record("yes"))
    no_button = widgets.Button(description="なし (0)", button_style="")
    no_button.on_click(lambda _: record("no"))
    unsure_button = widgets.Button(description="わからない", button_style="warning")
    unsure_button.on_click(lambda _: record("unsure"))

    display(
        widgets.HTML(
            f"<b>{LABEL_JA[label]}</b> が該当するかだけを判定してください。"
            "以前の答えは表示していません(引きずられないようにするため)。"
        ),
        progress_label,
        output,
        widgets.HBox([yes_button, no_button, unsure_button]),
    )
    show_current()
