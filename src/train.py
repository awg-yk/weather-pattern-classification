import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.dataset import WeatherMapDataset
from src.era5_grid import ERA5GridDataset, compute_grid_stats
from src.labels import LABELS
from src.model import DEFAULT_IMAGE_SIZE, build_model, save_checkpoint
from src.split import SPLIT_MODES, VAL_MODES, make_splits

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


def _macro_ap(targets, scores) -> float:
    """ラベルごとのaverage precisionを、そのデータに現れたラベルだけで平均する。

    1件も陽性が無いラベルのAPは定義できない(sklearnは0を返す)。検証データは
    300件足らずなので、希少ラベルが1件も出ない回がある。それを0として混ぜると
    「予測できなかった」のと区別がつかず、平均が不当に下がる。
    """
    scores_per_label = [
        average_precision_score(targets[:, i], scores[:, i])
        for i in range(targets.shape[1])
        if targets[:, i].sum() > 0
    ]
    return float(np.mean(scores_per_label)) if scores_per_label else float("nan")


def _chance_ap(targets) -> float:
    """当てずっぽうのmacro AP。ラベルごとのAPは、順位付けに情報が無ければ出現率に等しい。

    APの絶対値は出現率で決まるので、基準を出さずに見ると解釈を誤る。実際、
    学習データの出現率から出した基準を、季節が偏った検証セット(8〜12月)の
    val_apと比べて「学習できていない」と読み違えた。基準は必ず同じデータから取る。
    """
    prevalence = targets.mean(axis=0)
    present = prevalence > 0
    return float(prevalence[present].mean()) if present.any() else float("nan")


def run_epoch(model, loader, criterion, optimizer, device, train: bool, threshold: float = 0.5):
    """マルチラベル学習: labelsは各クラス0/1のmulti-hotベクトル、BCEで学習する。

    (損失, 完全一致率, macro F1, macro AP)を返す。

    完全一致率は「正解ラベル集合と予測ラベル集合が完全一致した割合」(subset accuracy)。
    macro F1はthresholdを固定して測るため、モデル選択には向かない -- pos_weightで
    再現寄りに学習させている以上、学習初期は何でも陽性と予測して0.5を超えやすく、
    学習が進んで予測が鋭くなるほど0.5超えが減る。順位付けが良くなっていても
    F1@0.5は下がりうる。最終評価は閾値を最適化して測るので、指標が食い違う。

    そこでmacro AP(ラベルごとのaverage precisionの平均)も返す。閾値を決めずに
    順位付けの良さだけを測るため、閾値最適化後の性能と素直に対応する。
    """
    model.train() if train else model.eval()
    total_loss, exact_match, total = 0.0, 0, 0
    all_preds, all_probs, all_targets = [], [], []

    with torch.set_grad_enabled(train):
        for images, features, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            # 要素数0なら「ERA5を使わない構成」なので、そのままNoneとして渡す
            features = features.to(device) if features.numel() else None

            if train:
                optimizer.zero_grad()
            outputs = model(images, features) if features is not None else model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            exact_match += (preds == labels).all(dim=1).sum().item()
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            all_preds.append(preds.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    targets = torch.cat(all_targets).numpy()
    macro_f1 = f1_score(
        targets, torch.cat(all_preds).numpy(), average="macro", zero_division=0,
    )
    return (total_loss / total, exact_match / total, macro_f1,
            _macro_ap(targets, torch.cat(all_probs).numpy()), _chance_ap(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="画像が入ったディレクトリ(chartモードで必須)")
    parser.add_argument("--labels", required=True, help="labels.csvのパス")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--era5-features",
        default=None,
        help="scripts/era5_features.py が出力したCSV。画像に加えてERA5の数値も使う"
        "(--input-mode era5-grid とは併用できない)",
    )
    parser.add_argument(
        "--input-mode",
        default="chart",
        choices=["chart", "era5-grid"],
        help="chart(既定)=天気図の画像を入力にする。"
        "era5-grid=ERA5の海面更正気圧・850hPa気温の格子を、圧縮せずそのまま"
        "画像の代わりに入力する(前線は含まれないため、前線系ラベルは原理的に不利)",
    )
    parser.add_argument(
        "--era5-grid-dir",
        default="data/raw/era5",
        help="--input-mode era5-grid のときの、ERA5 netCDFの置き場所",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=128,
        help="--input-mode era5-grid のときの、格子をリサイズする一辺の大きさ",
    )
    parser.add_argument(
        "--arch",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "small_cnn"],
        help="small_cnnはEfficientNet-B0の約17分の1(24万パラメータ)の小さな畳み込みネット。"
        "ERA5格子はImageNetの事前学習が効かずゼロから学習するため、EfficientNetでは"
        "学習データ千件あまりに対して大きすぎ、極端な過学習に陥る",
    )
    parser.add_argument(
        "--coordconv",
        action="store_true",
        help="入力に座標(緯度経度に相当)チャンネルを足す。位置で決まる気圧配置の判別を助ける",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="ImageNetの事前学習重みを使わずゼロから学習する。"
        "既定では、ERA5格子(--input-mode era5-grid)でも最初の畳み込みだけ作り直して"
        "事前学習重みを引き継ぐ。事前学習の寄与そのものを測りたいときに指定する",
    )
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
        "--val-mode",
        default="tail",
        choices=VAL_MODES,
        help="loyoでの検証データの取り方。tail=学習期間の末尾(既定)、"
        "spread=1年を通して等間隔に週を抜く。spreadは検証が通年になるが、"
        "抜いた週の前後を学習から除くぶん学習データが減る",
    )
    parser.add_argument(
        "--select-metric",
        choices=["macro_ap", "macro_f1", "val_loss"],
        default="macro_ap",
        help="ベストモデルの保存・早期終了の判定に使う指標。"
        "既定のmacro_apは閾値を決めずに順位付けの良さを測るので、閾値を最適化する"
        "最終評価と対応する。macro_f1は閾値0.5固定のため、pos_weightで再現寄りに"
        "学習させると初期エポックが最高値になりやすい(過去の実験の再現用に残している)",
    )
    args = parser.parse_args()

    # 再現性のため、乱数源をすべて固定する(cuDNNの非決定的アルゴリズムも無効化)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.input_mode == "chart" and not args.data_dir:
        raise SystemExit("chartモードでは --data-dir が必要です")
    if args.input_mode == "era5-grid" and args.era5_features:
        raise SystemExit(
            "--input-mode era5-grid と --era5-features は併用できません"
            "(era5-gridは格子そのものを入力にするため、26個への要約は不要)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.input_mode == "era5-grid":
        # 格子の読み込みはnetCDFの解凍を伴い重いため、train用・eval用で
        # 二重に読み込まない。1つ構築してから浅いコピーで augment だけ切り替える
        # (self.grids は同じ配列を参照するので、正規化の統計は両方に個別に設定する)。
        import copy

        train_base = ERA5GridDataset(
            args.labels, args.era5_grid_dir, years=args.years,
            grid_size=args.grid_size, augment=True,
        )
        eval_base = copy.copy(train_base)
        eval_base.augment = False
    else:
        # 学習用と検証用でtransformが異なるため、データセットの実体を2つ作る。
        # Subsetは元のデータセットへの「参照」を共有するので、1つのインスタンスを
        # 分割して片方のtransformだけ差し替えることはできない
        # (差し替えると学習側のデータ拡張まで消える)。
        train_base = WeatherMapDataset(
            args.data_dir,
            args.labels,
            transform=get_transforms(train=True, image_size=args.image_size),
            years=args.years,
            features_csv=args.era5_features,
        )
        eval_base = WeatherMapDataset(
            args.data_dir,
            args.labels,
            transform=get_transforms(train=False, image_size=args.image_size),
            years=args.years,
            features_csv=args.era5_features,
        )

    splits = make_splits(
        train_base.df,
        mode=args.split_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_year=args.test_year,
        gap_days=args.gap_days,
        val_mode=args.val_mode,
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
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        coordconv=args.coordconv,
        num_features=len(train_base.feature_cols),
        in_channels=2 if args.input_mode == "era5-grid" else 3,
        arch=args.arch,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"モデル: {args.arch} (学習するパラメータ {trainable:,}個 / 学習データ{len(train_ds)}件)")
    if args.no_pretrained:
        print("事前学習重みを使わずゼロから学習します(--no-pretrained)")
    if args.coordconv:
        print("CoordConv: 入力に座標チャンネルを追加します(位置に依存する気圧配置の判別用)")
    if train_base.feature_cols:
        # 正規化の統計は学習用サブセットだけから求める。
        # 検証・テストの分を含めると、そこにしか無い情報が学習側に漏れる。
        train_features = train_base.features[train_ds.indices]
        model.set_feature_stats(train_features.mean(0), train_features.std(0))
        print(f"ERA5特徴量{len(train_base.feature_cols)}個を併用します"
              f"(正規化は学習{len(train_features)}件から算出)")
    if args.input_mode == "era5-grid":
        # 正規化の統計も学習用サブセットだけから求める(理由は上と同じ)。
        # train_base/eval_baseは同じ格子配列を参照する2つのビューなので、両方に設定する。
        mean, std = compute_grid_stats(train_base, train_ds.indices)
        train_base.set_stats(mean, std)
        eval_base.set_stats(mean, std)
        print(f"ERA5格子(気圧・気温)を直接入力します"
              f"(grid_size={args.grid_size} / 正規化は学習{len(train_ds)}件から算出)")
        print(f"  気圧の平均/標準偏差: {mean[0]:.2f} / {std[0]:.2f}"
              f"  気温の平均/標準偏差: {mean[1]:.2f} / {std[1]:.2f}")

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

    # ベストモデルの選択基準。既定はmacro AP(run_epochを参照)。閾値0.5固定の
    # macro F1で選ぶと、pos_weightで陽性寄りに出力する初期エポックが最高値を取り、
    # 実質未学習の重みが保存されてしまう(3つのfoldで実際に起きた)。
    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # エポックごとの経過は、後から結果がおかしかったときの唯一の手掛かりになる
    # (学習側も低いのか=最適化の失敗、学習側だけ高いのか=過学習、で打つ手が違う)。
    # コンソールに出すだけだと流れて消えるため、重みの隣に残す。
    history = []
    history_path = out_path.with_suffix(".history.json")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, train_ap, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_f1, val_ap, val_chance_ap = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} train_ap={train_ap:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} "
            f"val_ap={val_ap:.4f}(基準{val_chance_ap:.3f}) lr={scheduler.get_last_lr()[0]:.2e}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "train_f1": train_f1, "train_ap": train_ap,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_f1": val_f1, "val_ap": val_ap, "val_chance_ap": val_chance_ap,
            "lr": scheduler.get_last_lr()[0],
        })
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        # 「大きいほど良い」に符号を揃えて一律に比較する
        score = {"macro_ap": val_ap, "macro_f1": val_f1}.get(args.select_metric, -val_loss)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            # 保存先は同じ"image_size"フィールドを使い回す(grid modeでは実質grid_size)。
            # 推論側がここに保存された値をそのまま前処理サイズとして使うため、
            # フィールド名を分けずに済ませている。
            saved_size = args.grid_size if args.input_mode == "era5-grid" else args.image_size
            save_checkpoint(out_path, model, image_size=saved_size)
            print(f"  saved best model to {out_path} ({args.select_metric}={abs(score):.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  {args.select_metric}が{args.patience}エポック改善しなかったため早期終了します")
                break

    # どのエポックが選ばれたかは、結果を読むときの前提になる。極端に早いエポックが
    # 選ばれていれば、それ以降の学習が汎化を損なっているということ(格子入力で実際に
    # 起きた: エポック1の重みがエポック24の重みよりテストで良かった)。
    print(f"\nベストはエポック {best_epoch}/{args.epochs} "
          f"({args.select_metric}={abs(best_score):.4f}) -- 経過は {history_path}")


if __name__ == "__main__":
    main()
