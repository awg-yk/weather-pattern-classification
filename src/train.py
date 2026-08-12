import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from src.dataset import WeatherMapDataset
from src.labels import LABELS
from src.model import build_model

IMAGE_SIZE = 224


def get_transforms(train: bool):
    # 天気図は左右反転すると「西高東低」が「東高西低」になるなど地理的な意味が
    # 壊れるため、水平反転は使わない。代わりに軽い明るさ/コントラストの変動と
    # 小さな回転・平行移動でデータ量の少なさを補う。
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        ops.append(transforms.RandomAffine(degrees=3, translate=(0.02, 0.02)))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
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
    (subset accuracy)を指標として使う。
    """
    model.train() if train else model.eval()
    total_loss, exact_match, total = 0.0, 0, 0

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

    return total_loss / total, exact_match / total


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = WeatherMapDataset(args.data_dir, args.labels, transform=get_transforms(train=True))
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = get_transforms(train=False)

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

    # 過学習の判定・モデル保存はval_lossを基準にする(小さな検証セットでは
    # 「完全一致率」は数枚のブレで大きく上下しやすく、あてにならないため)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_path)
            print(f"  saved best model to {out_path} (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  val_lossが{args.patience}エポック改善しなかったため早期終了します")
                break


if __name__ == "__main__":
    main()
