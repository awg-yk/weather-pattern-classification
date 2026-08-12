"""
Colab上で画像を1枚ずつ表示し、ボタンクリックでラベル付けする簡易ツール。

Colabのノートブックセルで以下のように使う:

    import sys
    sys.path.append("/content/weather-pattern-classification")
    from scripts.label_tool import run_labeling_session

    run_labeling_session(
        images_dir="data/processed/jma",
        labels_csv="data/labels.csv",
    )

- 既に data/labels.csv にラベル済みのファイルは自動的にスキップされる（中断・再開が可能）
- 「わからない/該当なし」ボタンで unclassified として記録し、後で見直せる
- 「戻る」ボタンで直前の1件を取り消せる
"""

import csv
from pathlib import Path

from IPython.display import display, clear_output
import ipywidgets as widgets
from PIL import Image

from src.labels import LABELS, LABEL_JA

SKIP_LABEL = "unclassified"
SKIP_LABEL_JA = "わからない/該当なし"


def _load_labeled_filenames(labels_csv: Path) -> set:
    if not labels_csv.exists():
        return set()
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["filename"] for row in reader}


def _append_label(labels_csv: Path, filename: str, label: str, source: str) -> None:
    file_exists = labels_csv.exists()
    with open(labels_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "label", "source", "date"])
        date_guess = filename.split("_")[-1].split(".")[0][:8] if "_" in filename else ""
        writer.writerow([filename, label, source, date_guess])


def _remove_last_label(labels_csv: Path) -> str | None:
    if not labels_csv.exists():
        return None
    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return None
    removed = rows.pop()
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return removed[0]


def run_labeling_session(images_dir: str, labels_csv: str, source: str = "jma_manual"):
    images_dir = Path(images_dir)
    labels_csv = Path(labels_csv)

    all_files = sorted(images_dir.glob("*.png"))
    labeled = _load_labeled_filenames(labels_csv)
    remaining = [f for f in all_files if f.name not in labeled]

    state = {"idx": 0}
    output = widgets.Output()

    progress_label = widgets.Label()

    def update_progress():
        done = len(all_files) - len(remaining) + state["idx"]
        progress_label.value = f"{done} / {len(all_files)} 完了 (残り {len(remaining) - state['idx']} 枚)"

    def show_current():
        with output:
            clear_output(wait=True)
            if state["idx"] >= len(remaining):
                print("すべてラベル付けが完了しました。")
                return
            img_path = remaining[state["idx"]]
            print(img_path.name)
            display(Image.open(img_path).resize((500, 375)))
        update_progress()

    def on_label_click(label):
        def handler(_):
            if state["idx"] >= len(remaining):
                return
            img_path = remaining[state["idx"]]
            _append_label(labels_csv, img_path.name, label, source)
            state["idx"] += 1
            show_current()
        return handler

    def on_back_click(_):
        if state["idx"] == 0:
            return
        removed = _remove_last_label(labels_csv)
        if removed:
            state["idx"] -= 1
        show_current()

    label_buttons = [
        widgets.Button(description=LABEL_JA[label], layout=widgets.Layout(width="220px"))
        for label in LABELS
    ]
    for btn, label in zip(label_buttons, LABELS):
        btn.on_click(on_label_click(label))

    skip_button = widgets.Button(description=SKIP_LABEL_JA, button_style="warning")
    skip_button.on_click(on_label_click(SKIP_LABEL))

    back_button = widgets.Button(description="戻る", button_style="danger")
    back_button.on_click(on_back_click)

    grid = widgets.GridBox(label_buttons, layout=widgets.Layout(grid_template_columns="repeat(2, 220px)"))
    controls = widgets.HBox([skip_button, back_button])

    display(progress_label, output, grid, controls)
    show_current()
