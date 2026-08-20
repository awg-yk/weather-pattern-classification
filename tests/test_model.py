"""モデルの組み立てと重みの保存・復元のテスト。

構成(CoordConvの有無・ERA5特徴量の数)が食い違ったまま重みを読むと、
黙って別のモデルとして動いてしまい、意味のない数値が出る。
"""

import copy

import pytest
import torch

from src.labels import LABELS
from src.model import CoordConv, backbone, build_model, load_model, save_checkpoint

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
