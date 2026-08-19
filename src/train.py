import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.dataset import WeatherMapDataset
from src.labels import LABELS
from src.model import DEFAULT_IMAGE_SIZE, build_model, save_checkpoint
from src.split import SPLIT_MODES, make_splits

IMAGE_SIZE = DEFAULT_IMAGE_SIZE


def get_transforms(train: bool, image_size: int = IMAGE_SIZE):
    # 天気図は左右反転すると「西高東低」が「東高西低」になるなど地理的な意味が
    # 壊れるため、水平反転は使わない。代わりに軽い明るさ/コントラストの変動と
    # 小さな回転・平行移動でデータ量の少なさを補う。
    # 前線の種別は色(温暖前線=赤・寒冷前線=青)が担っているため、色相は変えない。
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        ops.append(transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train:
        # 等圧線・前線の一部が欠けても分類できるよう、局所的な遮蔽への頑健性を上げる
        ops.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.08)))
    return transforms.Compose(ops)


def compute_pos_weight(train_subset, num_classes: int, cap: float = 20.0) -> torch.Tensor:
    """クラス不均衡対策のpos_weightを、学習用サブセットのラベル分布から算出する。

    出現頻度が低いラベルほど「陽性を見逃した時の損失」を大きくすることで、
    モデルが安全策で陰性ばかり予測するのを防ぐ。(負例数/正例数)で計算し、
    極端に少ないクラスで値が発散しないようcapで上限を設ける。
    画像を読み込まず、ラベルのDataFrameだけを見て計算する。
    """
    base_dataset = train_subset.dataset
    rows = base_dataset.df.iloc[train_subset.indices]

    pos_counts = torch.zeros(num_classes)
    for label in LABELS:
        idx = LABELS.index(label)
        pos_counts[idx] = rows["parsed_labels"].apply(lambda ls, l=label: l in ls).sum()

    total = len(rows)
    neg_counts = total - pos_counts
    pos_counts = pos_counts.clamp(min=1.0)
    weight = (neg_counts / pos_counts).clamp(max=cap)
    return weight


def run_epoch(model, loader, criterion, optimizer, device, train: bool, threshold: float = 0.5):
    """マルチラベル学習: labelsは各クラス0/1のmulti-hotベクトル、BCEで学習する。

    精度は「画像ごとに正解ラベル集合と予測ラベル集合が完全一致した割合」
    (subset accuracy)と、ラベルごとのF1を平均したmacro F1の両方を返す。
    完全一致率は数枚のブレで大きく上下するため、モデル選択にはmacro F1を使う。
    """
    model.train() if train else model.eval()
    total_loss, exact_match, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(outputs) > threshold).float()
            exact_match += (preds == labels).all(dim=1).sum().item()
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_preds.append(preds.detach().cpu())
            all_targets.append(labels.detach().cpu())

    macro_f1 = f1_score(
        torch.cat(all_targets).numpy(),
        torch.cat(all_preds).numpy(),
        average="macro",
        zero_division=0,
    )
    return total_loss / total, exact_match / total, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="画像が入ったディレクトリ")
    parser.add_argument("--labels", required=True, help="labels.csvのパス")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="特徴抽出部を凍結し分類ヘッドのみ学習する(データが少ない場合の過学習対策)",
    )
    parser.add_argument("--patience", type=int, default=8, help="val_lossがこの回数改善しなければ早期終了")
    parser.add_argument(
        "--pos-weight-cap",
        type=float,
        default=8.0,
        help="クラス不均衡対策のpos_weightの上限。大きいほど少数ラベルのrecallを稼ぐ代わりにprecisionが下がりやすい",
    )
    parser.add_argument("--out", default="weights/model.pt")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train/val分割の乱数シード。学習件数を変えて比較する際はこれを固定して"
        "検証セットを揃えること(--train-limitと併用する場合に特に重要)",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="学習に使う件数を制限する(検証セットのサイズは変えない)。"
        "学習件数と性能の関係を調べる実験用。例: 100, 300, 600 ...",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="モデルへの入力解像度。天気図は等圧線や前線記号が細かいため、"
        f"既定の{IMAGE_SIZE}では潰れやすい。384程度まで上げると精度が改善しうるが"
        "VRAMを解像度の2乗で消費するので--batch-sizeを下げること",
    )
    parser.add_argument(
        "--split-mode",
        choices=list(SPLIT_MODES),
        default="temporal",
        help="train/val/testの分け方。天気図は隣り合う日どうしが酷似しているため、"
        "randomだと検証画像と1日違いの画像で学習してしまいスコアが水増しされる。"
        "既定は時間ブロック分割。randomは過去の実験を再現する用途のみ",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="最終報告用に取り分けるテストデータの割合。学習にも閾値探索にも使わない。"
        "0だとテストセットを作らない(従来互換)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="対象にする年(例: --years 2023 2024 2025)。指定しなければ全期間",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=None,
        help="--split-mode loyo のとき、テストに回す年",
    )
    parser.add_argument(
        "--gap-days",
        type=int,
        default=3,
        help="loyoで、テスト年からこの日数以内の学習データを除外する(年境界のリーク対策)",
    )
    parser.add_argument(
        "--select-metric",
        choices=["macro_f1", "val_loss"],
        default="macro_f1",
        help="ベストモデルの保存・早期終了の判定に使う指標。"
        "val_lossは改善が早く止まりF1のピークを取り逃すことが実測で確認されているため、"
        "既定はmacro_f1(過去の実験を再現する場合のみval_lossを指定する)",
    )
    args = parser.parse_args()

    # 再現性のため、乱数源をすべて固定する(cuDNNの非決定的アルゴリズムも無効化)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習用と検証用でtransformが異なるため、データセットの実体を2つ作る。
    # Subsetは元のデータセットへの「参照」を共有するので、1つのインスタンスを
    # 分割して片方のtransformだけ差し替えることはできない
    # (差し替えると学習側のデータ拡張まで消える)。
    train_base = WeatherMapDataset(
        args.data_dir,
        args.labels,
        transform=get_transforms(train=True, image_size=args.image_size),
        years=args.years,
    )
    eval_base = WeatherMapDataset(
        args.data_dir,
        args.labels,
        transform=get_transforms(train=False, image_size=args.image_size),
        years=args.years,
    )

    splits = make_splits(
        train_base.df,
        mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_year=args.test_year,
        gap_days=args.gap_days,
    )
    train_ds = Subset(train_base, splits["train"])
    val_ds = Subset(eval_base, splits["val"])
    # テストセットはここでは一切触らない。src/evaluate.py --split test で
    # 同じ分割を復元し、最終報告のときに1回だけ測る。

    if args.train_limit is not None:
        if args.train_limit > len(train_ds):
            raise ValueError(
                f"--train-limit={args.train_limit} が学習用データ件数({len(train_ds)})を超えています"
            )
        # --seedを揃えれば同じ検証セットのまま学習サブセットだけ縮小できる。
        # compute_pos_weight()はtrain_ds.dataset(元のWeatherMapDataset)を
        # 前提にしているため、Subsetを二重に重ねず単一階層のSubsetに作り直す。
        limit_generator = torch.Generator().manual_seed(args.seed)
        picked = torch.randperm(len(train_ds), generator=limit_generator)[: args.train_limit].tolist()
        limited_indices = [train_ds.indices[i] for i in picked]
        train_ds = Subset(train_ds.dataset, limited_indices)
        print(f"--train-limit指定: 学習データを{args.train_limit}件に制限します")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(
        num_classes=len(LABELS),
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)

    pos_weight = compute_pos_weight(train_ds, num_classes=len(LABELS), cap=args.pos_weight_cap).to(device)
    print("pos_weight:", {label: round(w, 2) for label, w in zip(LABELS, pos_weight.tolist())})
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # 学習率を徐々に下げることで、終盤の細かい収束と汎化性能を安定させる
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ベストモデルの選択基準。val_lossは早い段階で改善が止まる一方、その後も
    # macro F1が伸び続けることが実測で確認されているため、既定はmacro F1。
    # (--select-metric val_loss で従来の挙動に戻せる)
    best_score = float("-inf")
    epochs_without_improvement = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # 「大きいほど良い」に符号を揃えて一律に比較する
        score = val_f1 if args.select_metric == "macro_f1" else -val_loss
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(out_path, model, image_size=args.image_size)
            print(f"  saved best model to {out_path} ({args.select_metric}={abs(score):.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  {args.select_metric}が{args.patience}エポック改善しなかったため早期終了します")
                break


if __name__ == "__main__":
    main()
