"""
Grad-CAMでモデルが画像のどこに注目して予測したかを可視化する。

モデルが本当にH/Lの記号・前線・等圧線の形を手がかりにしているのか、
それとも無関係な部分(枠の残りカスなど)を見て偶然当てているだけなのかを
確認するために使う。

Colabのノートブックセルで以下のように使う:

    import sys
    sys.path.append("/content/weather-pattern-classification")
    from scripts.gradcam import show_gradcam

    show_gradcam(
        image_path="/content/drive/MyDrive/.../some_chart.png",
        weights_path="/content/drive/MyDrive/weather-pattern-classification-data/weights/model.pt",
        top_k=3,          # 確信度が高い上位k個のラベルについて可視化する
        apply_preprocess=True,  # 生のJMA画像(枠・スタンプ付き)ならTrue、前処理済みならFalse
    )
"""

import subprocess
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def _register_cjk_font() -> bool:
    """OSのfontconfigから直接CJKフォントファイルを探して matplotlib に登録する。

    matplotlibは起動時にフォント一覧をキャッシュしているため、実行中に
    `apt-get install fonts-noto-cjk` してもキャッシュが更新されず
    ttflistに反映されないことがある。fc-list(fontconfig)はOS側の情報を
    直接見るので、その場でインストールしたフォントもすぐ拾える。
    """
    try:
        result = subprocess.run(
            ["fc-list", ":lang=ja", "file"], capture_output=True, text=True, timeout=10
        )
        font_paths = [line.split(":")[0].strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        font_paths = []

    for path in font_paths:
        try:
            fm.fontManager.addfont(path)
        except Exception:
            continue

    if not font_paths:
        return False

    try:
        font_name = fm.FontProperties(fname=font_paths[0]).get_name()
        matplotlib.rcParams["font.family"] = font_name
        return True
    except Exception:
        return False


if not _register_cjk_font():
    print(
        "警告: 日本語フォントが見つかりませんでした。"
        "`!apt-get -qq install -y fonts-noto-cjk` を実行してから、"
        "このセルを再実行(ランタイム再起動は不要)してください。"
    )

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.preprocess_jma import DEFAULT_STAMP_BOX, autocrop_to_content, mask_stamp_box
from src.labels import INDEX_TO_LABEL, LABEL_JA, LABELS
from src.model import build_model
from src.train import get_transforms


class GradCAM:
    """EfficientNet-B0の最終畳み込み層(model.features)にフックを仕込んでCAMを計算する。"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """input_tensor: (1, C, H, W)。指定クラスのロジットに対するCAMを返す(0-1に正規化済み)。"""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[0, class_idx]
        score.backward()

        # チャンネルごとの勾配の平均を重みとし、活性化マップを加重和する(Grad-CAMの定義)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam[0, 0].cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def _load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(num_classes=len(LABELS))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _overlay_heatmap(base_image: Image.Image, cam: np.ndarray, max_alpha: float = 0.6) -> Image.Image:
    """CAMの強さに応じて透明度を変えながら重ねる。

    一律の透明度で塗ると、H/Lの記号や気圧の数値のような細かい文字が
    色に埋もれて読めなくなる。注目度がほぼ0の場所は元の線画をそのまま残し、
    注目度が高い場所だけ強く色を乗せることで、文字の判読性を保つ。
    """
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(base_image.size, Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0  # (H, W), 0-1

    heatmap = plt.cm.jet(cam_resized)[:, :, :3] * 255.0  # (H, W, 3)
    base_arr = np.array(base_image.convert("RGB")).astype(np.float32)

    alpha_map = (cam_resized * max_alpha)[:, :, None]  # (H, W, 1), 場所ごとの透明度
    overlay = (1 - alpha_map) * base_arr + alpha_map * heatmap
    return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))


def explain_top_prediction(image_path: str, weights_path: str, apply_preprocess: bool = True):
    """最も確信度が高いラベルについてだけヒートマップ画像を作り、

    (前処理後の元画像, 最上位ラベルのヒートマップ画像, [(ラベル, 確信度), ...確信度降順]) を返す。
    predict.ipynbのように「1位だけ画像、残りはテキストでよい」という用途向け。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(weights_path, device)
    gradcam = GradCAM(model)

    raw_image = Image.open(image_path).convert("RGB")
    display_image = raw_image
    if apply_preprocess:
        display_image = autocrop_to_content(display_image)
        display_image = mask_stamp_box(display_image, DEFAULT_STAMP_BOX)

    transform = get_transforms(train=False)
    input_tensor = transform(display_image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0].cpu()
    sorted_indices = torch.argsort(probs, descending=True).tolist()
    ranked = [(INDEX_TO_LABEL[i], probs[i].item()) for i in sorted_indices]

    top_idx = sorted_indices[0]
    cam = gradcam.generate(input_tensor, top_idx)
    top_overlay = _overlay_heatmap(display_image, cam)

    return display_image, top_overlay, ranked


def show_gradcam(
    image_path: str,
    weights_path: str,
    top_k: int = 3,
    apply_preprocess: bool = True,
    figsize_per_panel: float = 4.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(weights_path, device)
    gradcam = GradCAM(model)

    raw_image = Image.open(image_path).convert("RGB")
    display_image = raw_image
    if apply_preprocess:
        display_image = autocrop_to_content(display_image)
        display_image = mask_stamp_box(display_image, DEFAULT_STAMP_BOX)

    transform = get_transforms(train=False)
    input_tensor = transform(display_image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(input_tensor))[0].cpu()
    top_indices = torch.argsort(probs, descending=True)[:top_k].tolist()

    fig, axes = plt.subplots(1, top_k + 1, figsize=(figsize_per_panel * (top_k + 1), figsize_per_panel))
    axes[0].imshow(display_image)
    axes[0].set_title("入力画像(前処理後)")
    axes[0].axis("off")

    for ax, idx in zip(axes[1:], top_indices):
        cam = gradcam.generate(input_tensor, idx)
        overlaid = _overlay_heatmap(display_image, cam)
        label = INDEX_TO_LABEL[idx]
        ax.imshow(overlaid)
        ax.set_title(f"{LABEL_JA[label]}\n({probs[idx].item() * 100:.1f}%)")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
