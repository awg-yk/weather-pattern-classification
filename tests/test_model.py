"""モデルの組み立てと重みの保存・復元のテスト。

構成(CoordConvの有無・ERA5特徴量の数)が食い違ったまま重みを読むと、
黙って別のモデルとして動いてしまい、意味のない数値が出る。
"""

import copy
import pathlib
import tempfile

import numpy as np
import pytest
import torch

from src.labels import LABELS
from src.model import (
    CoordConv,
    _adapt_first_conv,
    backbone,
    build_model,
    load_model,
    save_checkpoint,
)

IMAGE = torch.randn(2, 3, 64, 64)


def test_coordconv_starts_out_identical_to_the_plain_model():
    """座標チャンネル分の重みは0で始まる。だから導入しても悪化しない、と言える。"""
    torch.manual_seed(0)
    plain = build_model(pretrained=False)
    reference = copy.deepcopy(plain).eval()
    wrapped = CoordConv(plain).eval()
    assert torch.allclose(reference(IMAGE), wrapped(IMAGE), atol=1e-6)


def test_coordconv_feeds_normalised_coordinates():
    model = build_model(pretrained=False, coordconv=True).eval()
    captured = {}
    backbone(model).features[0][0].register_forward_hook(
        lambda module, inputs, output: captured.update(x=inputs[0])
    )
    model(torch.zeros(1, 3, 8, 8))
    x = captured["x"]
    assert x.shape[1] == 5, "座標2チャンネルが足されていない"
    assert x[0, 3, 0, 0] == pytest.approx(-1.0)   # 上端
    assert x[0, 3, -1, 0] == pytest.approx(1.0)   # 下端
    assert x[0, 4, 0, 0] == pytest.approx(-1.0)   # 左端
    assert x[0, 4, 0, -1] == pytest.approx(1.0)   # 右端


@pytest.mark.parametrize("coordconv", [False, True])
@pytest.mark.parametrize("num_features", [0, 5])
def test_checkpoint_round_trip(tmp_path, coordconv, num_features):
    torch.manual_seed(0)
    model = build_model(pretrained=False, coordconv=coordconv, num_features=num_features).eval()
    features = torch.randn(2, num_features) if num_features else None
    if num_features:
        model.set_feature_stats(features.mean(0), features.std(0))

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, image_size=64)
    restored, meta = load_model(path)
    restored.eval()

    assert meta["coordconv"] is coordconv
    assert meta["num_features"] == num_features
    assert meta["labels"] == list(LABELS)
    expected = model(IMAGE, features) if num_features else model(IMAGE)
    actual = restored(IMAGE, features) if num_features else restored(IMAGE)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_loading_into_a_mismatched_model_is_refused(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, build_model(pretrained=False, coordconv=True), image_size=64)

    from src.model import load_checkpoint

    with pytest.raises(ValueError, match="一致しません"):
        load_checkpoint(path, build_model(pretrained=False, coordconv=False))


def test_fusion_model_refuses_to_run_without_features():
    model = build_model(pretrained=False, num_features=5).eval()
    with pytest.raises(ValueError, match="ERA5"):
        model(IMAGE)


def test_feature_normalisation_uses_the_supplied_statistics():
    """正規化の統計は学習データだけから求める。ここが効いていないと、
    検証・テストの情報が学習側に漏れても気づけない。"""
    model = build_model(pretrained=False, num_features=3).eval()
    mean, std = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([2.0, 4.0, 6.0])
    model.set_feature_stats(mean, std)
    assert torch.allclose(model.feature_mean, mean)
    assert torch.allclose(model.feature_std, std)

    # 標準偏差0の特徴でも0除算にならない
    model.set_feature_stats(mean, torch.zeros(3))
    assert (model.feature_std > 0).all()


def test_non_rgb_input_keeps_the_pretrained_weights_past_the_first_layer():
    """ERA5格子(2ch)でも、形が合わないのは最初の畳み込みだけ。

    ここが壊れて事前学習重みを丸ごと捨てるようになると、精度は下がるが
    エラーにはならないので、テストしないと気づけない。
    """
    reference = build_model(pretrained=False)
    adapted = copy.deepcopy(reference)
    _adapt_first_conv(adapted, 2, pretrained=True)

    assert backbone(adapted).features[0][0].in_channels == 2
    # 2層目以降はそのまま引き継がれている
    for (name, before), (_, after) in zip(
        backbone(reference).features[1:].named_parameters(),
        backbone(adapted).features[1:].named_parameters(),
    ):
        assert torch.equal(before, after), name


def test_adapted_first_layer_preserves_the_scale_of_the_activations():
    """全チャンネルに同じ値を入れたとき、3ch版と同じ出力になるようスケールを合わせる。

    これを外すと活性が入力チャンネル数に比例して膨らみ、後続の層が前提と
    している大きさから外れる。
    """
    reference = build_model(pretrained=False).eval()
    adapted = copy.deepcopy(reference)
    _adapt_first_conv(adapted, 2, pretrained=True)
    adapted.eval()

    plane = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        expected = backbone(reference).features[0][0](plane.expand(-1, 3, -1, -1))
        actual = backbone(adapted).features[0][0](plane.expand(-1, 2, -1, -1))
    assert torch.allclose(expected, actual, atol=1e-5)


def test_grid_weights_survive_the_save_and_load_round_trip():
    """2チャンネルの重みを、CoordConvと併用しても正しく読み戻せること。"""
    model = build_model(pretrained=False, in_channels=2, coordconv=True).eval()
    grid = torch.randn(2, 2, 64, 64)
    with torch.no_grad():
        expected = model(grid)

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.pt"
    save_checkpoint(path, model, image_size=64)
    restored, _ = load_model(path, map_location="cpu")
    with torch.no_grad():
        assert torch.allclose(expected, restored.eval()(grid), atol=1e-6)


def test_macro_ap_ignores_labels_with_no_positives():
    """検証データに1件も現れないラベルは平均から外す。

    average_precision_scoreは陽性0のとき0を返すため、含めると「予測できなかった」
    のと区別がつかない。検証は300件足らずで、希少ラベルが1件も出ない回がある。
    """
    from src.train import _macro_ap

    targets = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    scores = np.array([[0.9, 0.5], [0.1, 0.5], [0.8, 0.5]])
    # 1列目だけで平均される -- 完全に順位付けできているので1.0
    assert _macro_ap(targets, scores) == pytest.approx(1.0)
    # どのラベルにも陽性が無ければ平均のとりようがない
    assert _macro_ap(np.zeros((3, 2)), scores) != _macro_ap(np.zeros((3, 2)), scores)


def test_macro_ap_rises_as_the_ranking_improves_where_f1_at_a_fixed_threshold_stalls():
    """モデル選択の指標をmacro F1からmacro APに変えた理由そのもの。

    pos_weightで陽性寄りに学習させるため、初期は何でも0.5を超えてF1が高く出る。
    順位付けが良くなっても出力が0.5未満に寄ればF1@0.5は伸びない。
    """
    from sklearn.metrics import f1_score

    from src.train import _macro_ap

    # 2列にするのは、1列だとsklearnが多クラス扱いにして陰性側のF1まで平均に
    # 入れてしまい、マルチラベルのmacro F1にならないため。
    targets = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    # 順位は完璧だが、確率はすべて0.5未満(閾値0.5では1件も陽性と判定されない)
    sharp = np.array([[0.45, 0.45], [0.40, 0.40], [0.10, 0.10], [0.05, 0.05]])
    assert _macro_ap(targets, sharp) == pytest.approx(1.0)
    assert f1_score(targets, (sharp > 0.5).astype(float), average="macro", zero_division=0) == 0.0


def test_small_cnn_beats_efficientnet_at_a_comparable_size():
    """勝敗を分けたのは容量ではなく構造だった、という測定結果を固定する。

    当初はEfficientNet-B0が大きすぎるのが原因と考えたが、幅を振ると性能は
    385万パラメータまで単調に上がった。既定の幅はEfficientNet-B0とほぼ同じ規模で、
    それでいてmacro F1は0.497対0.234(自明な予測は0.258)。容量を揃えて比べられる
    ことがこの既定値の意味なので、既定を小さい方に戻すとその対比が崩れる。
    """
    def count(model):
        return sum(p.numel() for p in model.parameters())

    small = count(build_model(pretrained=False, in_channels=2, arch="small_cnn"))
    efficientnet = count(build_model(pretrained=False, in_channels=2))
    assert 0.8 < small / efficientnet < 1.25

    # 幅で容量を振れること(6万〜600万を実際に走らせて頂点を探した)
    narrow = count(build_model(pretrained=False, in_channels=2, arch="small_cnn",
                               cnn_widths=[16, 32, 64, 64]))
    assert narrow * 10 < small


def test_small_cnn_accepts_any_input_size():
    """大域平均プーリングのため、格子サイズを変えても組み直さずに動くこと。"""
    model = build_model(pretrained=False, in_channels=2, arch="small_cnn").eval()
    for size in (64, 128, 224):
        with torch.no_grad():
            assert model(torch.randn(1, 2, size, size)).shape == (1, len(LABELS))


def test_small_cnn_survives_the_save_and_load_round_trip_with_coordconv():
    """archは重みに記録され、読み込み側が構成を知らなくても復元できること。

    ここが欠けると、EfficientNetの形で組んだモデルにSmallCNNの重みを読もうとして
    落ちるか、最悪は別物として黙って動く。
    """
    model = build_model(pretrained=False, in_channels=2, arch="small_cnn", coordconv=True).eval()
    grid = torch.randn(2, 2, 64, 64)
    with torch.no_grad():
        expected = model(grid)

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.pt"
    save_checkpoint(path, model, image_size=64)
    assert torch.load(path, map_location="cpu")["arch"] == "small_cnn"
    restored, _ = load_model(path, map_location="cpu")
    with torch.no_grad():
        assert torch.allclose(expected, restored.eval()(grid), atol=1e-6)


def test_small_cnn_widths_round_trip_through_the_checkpoint():
    """幅が違えば別の形なので、重みに記録して読み戻せること。

    記録し忘れると、既定の幅で組んだモデルに別の幅の重みを読もうとして落ちる。
    容量を振って比較する以上、幅の違う重みが並んで存在することになる。
    """
    model = build_model(pretrained=False, in_channels=2, arch="small_cnn",
                        cnn_widths=[16, 32, 64, 64], coordconv=True).eval()
    grid = torch.randn(2, 2, 64, 64)
    with torch.no_grad():
        expected = model(grid)

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.pt"
    save_checkpoint(path, model, image_size=64)
    assert torch.load(path, map_location="cpu")["cnn_widths"] == [16, 32, 64, 64]
    restored, _ = load_model(path, map_location="cpu")
    with torch.no_grad():
        assert torch.allclose(expected, restored.eval()(grid), atol=1e-6)
