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

import matplotlib.pyplot as plt
import numpy as np

from src.jp_font import missing_font_hint, register_matplotlib_cjk

if not register_matplotlib_cjk():
    print(
        "警告: 日本語フォントが見つかりませんでした。図中の日本語が豆腐(□)になります。"
        f"{missing_font_hint()}"
    )

import torch
import torch.nn.functional as F
from PIL import Image

from scripts.preprocess_jma import DEFAULT_STAMP_BOX, autocrop_to_content, mask_stamp_box
from src import calibration as calib
from src.labels import INDEX_TO_LABEL, LABEL_JA
from src.model import backbone, load_model
from src.regions import attention_mass, load_regions
from src.train import get_transforms


class GradCAM:
    """EfficientNet-B0の最終畳み込み層(model.features)にフックを仕込んでCAMを計算する。"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        # CoordConvで包まれている場合は中のEfficientNetを取り出す
        target_layer = backbone(model).features[-1]
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


def _load_model(weights_path: str, device: torch.device):
    """モデルと、その重みが前提とする入力サイズなどのメタデータを返す。"""
    model, meta = load_model(weights_path, map_location=device)
    model.to(device)
    model.eval()
    return model, meta


def _probabilities(model, input_tensor, weights_path, calibration=None):
    """校正済みの確信度を返す。校正ファイルが無ければ生の値のまま。

    生のsigmoid出力は学習時のpos_weightのぶん高く出るうえ、そのかさ上げ量が
    ラベルごとに違う。校正しないと、ヒートマップに添える%だけでなく
    「上位k件」の並び順まで少数ラベル寄りに歪む(src/calibration.py)。

    calibration を渡すとそれを使う(呼び出し側が既に読み込んでいる場合や、
    校正前の値をあえて見たい場合)。省略時は重みの隣の校正ファイルを探す。
    """
    with torch.no_grad():
        logits = model(input_tensor)[0].cpu().numpy()
    if calibration is None:
        calibration = calib.load_for_weights(weights_path, verbose=False)
    return torch.tensor(calibration.probabilities(logits), dtype=torch.float32)


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


def explain_top_predictions(
    image_path: str, weights_path: str, top_k: int = 3, apply_preprocess: bool = True,
    calibration=None,
):
    """確信度が高い上位top_k件についてヒートマップ画像を作り、

    (前処理後の元画像, [(ラベル, 確信度, ヒートマップ画像), ...上位top_k件],
     [(ラベル, 確信度), ...全ラベル確信度降順]) を返す。
    predict.ipynbのように「上位k件だけ画像、残りはテキストでよい」という用途向け。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = _load_model(weights_path, device)
    gradcam = GradCAM(model)

    raw_image = Image.open(image_path).convert("RGB")
    display_image = raw_image
    if apply_preprocess:
        display_image = autocrop_to_content(display_image)
        display_image = mask_stamp_box(display_image, DEFAULT_STAMP_BOX)

    transform = get_transforms(train=False, image_size=meta["image_size"])
    input_tensor = transform(display_image).unsqueeze(0).to(device)

    probs = _probabilities(model, input_tensor, weights_path, calibration)
    sorted_indices = torch.argsort(probs, descending=True).tolist()
    ranked = [(INDEX_TO_LABEL[i], probs[i].item()) for i in sorted_indices]

    top_overlays = []
    for idx in sorted_indices[:top_k]:
        cam = gradcam.generate(input_tensor, idx)
        overlay = _overlay_heatmap(display_image, cam)
        top_overlays.append((INDEX_TO_LABEL[idx], probs[idx].item(), overlay))

    return display_image, top_overlays, ranked


def explain_predictions_above_threshold(
    image_path: str, weights_path: str, threshold: float = None, apply_preprocess: bool = True,
    calibration=None,
):
    """確信度がthresholdを超えたラベルすべてについてヒートマップ画像を作る。

    threshold を省略すると、校正ファイルに入っているラベルごとのしきい値を使う
    (校正ファイルが無ければ一律0.5)。校正後の確率は少数ラベルほど小さい値に
    収まるため、一律0.5のままだとそれらが一切表示されなくなる。

    件数は固定せず、しきい値を超えた分だけ動的に変わる(0件になることもある)。
    (前処理後の元画像, [(ラベル, 確信度, ヒートマップ画像), ...しきい値超え・確信度降順],
     [(ラベル, 確信度), ...全ラベル確信度降順]) を返す。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = _load_model(weights_path, device)
    gradcam = GradCAM(model)

    raw_image = Image.open(image_path).convert("RGB")
    display_image = raw_image
    if apply_preprocess:
        display_image = autocrop_to_content(display_image)
        display_image = mask_stamp_box(display_image, DEFAULT_STAMP_BOX)

    transform = get_transforms(train=False, image_size=meta["image_size"])
    input_tensor = transform(display_image).unsqueeze(0).to(device)

    probs = _probabilities(model, input_tensor, weights_path, calibration)
    sorted_indices = torch.argsort(probs, descending=True).tolist()
    ranked = [(INDEX_TO_LABEL[i], probs[i].item()) for i in sorted_indices]

    if calibration is None:
        calibration = calib.load_for_weights(weights_path, verbose=False)

    overlays = []
    for idx in sorted_indices:
        label_threshold = (
            threshold if threshold is not None else calibration[INDEX_TO_LABEL[idx]].threshold
        )
        if probs[idx].item() <= label_threshold:
            # 確率の降順に見ているが、しきい値はラベルごとに違うので打ち切らず次を見る
            continue
        cam = gradcam.generate(input_tensor, idx)
        overlay = _overlay_heatmap(display_image, cam)
        overlays.append((INDEX_TO_LABEL[idx], probs[idx].item(), overlay))

    return display_image, overlays, ranked


def show_gradcam(
    image_path: str,
    weights_path: str,
    top_k: int = 3,
    apply_preprocess: bool = True,
    figsize_per_panel: float = 4.0,
    calibration=None,
    show_regions: bool = False,
):
    """上位top_k件のヒートマップを並べて表示する。

    show_regions=True にすると、そのラベルの「見るべき領域」(data/regions.csv)を
    枠で重ね、熱が枠内にどれだけ入っているかを見出しに出す。モデルの注目と
    教師データ側の想定を、同じ天気図の上で見比べるためのもの(src/regions.py)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = _load_model(weights_path, device)
    gradcam = GradCAM(model)

    raw_image = Image.open(image_path).convert("RGB")
    display_image = raw_image
    if apply_preprocess:
        display_image = autocrop_to_content(display_image)
        display_image = mask_stamp_box(display_image, DEFAULT_STAMP_BOX)

    transform = get_transforms(train=False, image_size=meta["image_size"])
    input_tensor = transform(display_image).unsqueeze(0).to(device)

    probs = _probabilities(model, input_tensor, weights_path, calibration)
    top_indices = torch.argsort(probs, descending=True)[:top_k].tolist()
    regions = load_regions() if show_regions else {}

    fig, axes = plt.subplots(1, top_k + 1, figsize=(figsize_per_panel * (top_k + 1), figsize_per_panel))
    axes[0].imshow(display_image)
    axes[0].set_title("入力画像(前処理後)")
    axes[0].axis("off")

    for ax, idx in zip(axes[1:], top_indices):
        cam = gradcam.generate(input_tensor, idx)
        overlaid = _overlay_heatmap(display_image, cam)
        label = INDEX_TO_LABEL[idx]
        ax.imshow(overlaid)
        title = f"{LABEL_JA[label]}\n({probs[idx].item() * 100:.1f}%)"
        region = regions.get(label)
        if region is not None:
            width, height = display_image.size
            left, top, right, bottom = region.pixel_box(width, height)
            ax.add_patch(
                plt.Rectangle((left, top), right - left, bottom - top,
                              linewidth=2.0, edgecolor="white", facecolor="none")
            )
            mass = attention_mass(cam, region)
            title += f"\n枠内 {mass:.0%} / 面積 {region.area:.0%}"
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
