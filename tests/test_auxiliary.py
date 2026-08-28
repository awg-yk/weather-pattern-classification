"""補助の答え(検出した数値を学習の答えに加える)のテスト。

守っているのは「**壊れても正常に見えるもの**」(tests/README.md)。
補助の損失が効いていなくても学習は最後まで走り、それらしい数字が出る。
行がずれていても同じで、別の日の位置を答えとして教えたまま完走する。

なぜこの仕組みが要るのか
------------------------
位置を**入力**として渡す CoordConv は既に試して効かなかった
(`docs/2026-08-25-attention-regions.md`)。入力は無視できるが、答えさせ
られる項目は無視できない、というのがこちらの狙いである。
"""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.dataset import WeatherMapDataset, compute_aux_stats
from src.labels import LABELS
from src.model import AuxiliaryTargets, build_model, load_checkpoint, load_model, save_checkpoint
from src.train import masked_mse


# --- 欠測を飛ばす損失 ----------------------------------------------------

def test_masked_mse_ignores_missing_values():
    """値の無い要素は損失に入れないこと。

    高気圧が1つも無い日に「高気圧の位置」は存在しない。0で埋めると
    「図の左上に高気圧がある」と教えることになる。
    """
    pred = torch.tensor([[1.0, 5.0]])
    target = torch.tensor([[1.0, float("nan")]])
    assert masked_mse(pred, target).item() == pytest.approx(0.0)


def test_masked_mse_is_zero_but_differentiable_when_all_missing():
    """1件も値が無くても、勾配のつながりが切れないこと。

    ここで定数0を返すと backward() が落ちる。
    """
    pred = torch.zeros(2, 3, requires_grad=True)
    loss = masked_mse(pred, torch.full((2, 3), float("nan")))
    loss.backward()
    assert loss.item() == 0.0
    assert pred.grad is not None


def test_masked_mse_averages_only_the_valid_entries():
    pred = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[2.0, float("nan"), 4.0]])
    assert masked_mse(pred, target).item() == pytest.approx((4 + 16) / 2)


# --- 出力の切り分け ------------------------------------------------------

def test_the_model_still_returns_only_the_ten_labels():
    """`model(x)` は10ラベルぶんだけ返すこと。

    **ここが崩れると、推論・評価・Grad-CAM・混合のすべてに補助の列が
    紛れ込む。**しかもエラーにはならず、余った列を確率として読んでしまう。
    """
    model = build_model(pretrained=False, num_aux=7).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, len(LABELS))


def test_forward_with_aux_splits_the_two_halves():
    model = build_model(pretrained=False, num_aux=7).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits, aux = model.forward_with_aux(x)
        whole = model.net(x)
    assert logits.shape == (2, len(LABELS)) and aux.shape == (2, 7)
    assert torch.allclose(logits, whole[:, :len(LABELS)])
    assert torch.allclose(aux, whole[:, len(LABELS):])


@pytest.mark.parametrize("kwargs", [
    {"num_aux": 5},
    {"num_aux": 5, "coordconv": True},
    {"num_aux": 4, "num_features": 6},
    {"num_aux": 3, "arch": "small_cnn", "cnn_widths": (8, 16)},
])
def test_the_checkpoint_round_trips(tmp_path, kwargs):
    """補助の本数を記録しないと、出力の形が変わって読み戻せない。

    ラッパーは入れ子になる(AuxiliaryTargets が FeatureFusion を包む)ので、
    isinstance だけで判定すると内側の設定を保存し損ねる。
    """
    torch.manual_seed(0)
    model = build_model(pretrained=False, **kwargs).eval()
    x = torch.randn(2, 3, 64, 64)
    features = torch.randn(2, 6) if kwargs.get("num_features") else None
    with torch.no_grad():
        expected = model(x, features) if features is not None else model(x)

    path = tmp_path / "m.pt"
    save_checkpoint(path, model, image_size=64)
    restored, meta = load_model(path)
    restored.eval()
    with torch.no_grad():
        got = restored(x, features) if features is not None else restored(x)
    assert meta["num_aux"] == kwargs["num_aux"]
    assert torch.allclose(got, expected, atol=1e-6)


def test_a_mismatched_aux_count_is_refused(tmp_path):
    """補助の本数が食い違う重みを、黙って読み込まないこと。"""
    model = build_model(pretrained=False, num_aux=5)
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, image_size=64)
    with pytest.raises(ValueError, match="num_aux"):
        load_checkpoint(path, build_model(pretrained=False))


# --- 本当に学習が進むか --------------------------------------------------

def test_the_auxiliary_head_actually_learns():
    """補助の損失で勾配が流れ、補助の誤差が下がること。

    **配線が切れていても学習は完走し、10ラベル側の数字は出る。**
    補助が効いていないことに気づけないので、ここで確かめる。
    画像の平均輝度を当てる、という自明な課題を与えている。
    """
    torch.manual_seed(0)
    model = build_model(pretrained=False, num_aux=1, arch="small_cnn",
                        cnn_widths=(8, 16))
    images = torch.randn(16, 3, 32, 32)
    aux_target = images.mean(dim=(1, 2, 3), keepdim=True).reshape(16, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    def aux_loss():
        return masked_mse(model.forward_with_aux(images)[1], aux_target)

    before = aux_loss().item()
    for _ in range(30):
        optimizer.zero_grad()
        aux_loss().backward()
        optimizer.step()
    after = aux_loss().item()
    assert after < before * 0.5, f"補助の誤差が下がっていない: {before:.4f} -> {after:.4f}"


# --- データセット側 ------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    images = tmp_path / "img"
    images.mkdir()
    names = []
    for i in range(6):
        name = f"Js_20240{i + 1}0100.png"
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(images / name)
        names.append(name)
    pd.DataFrame({"filename": names, "label": [LABELS[0]] * 6,
                  "date": [n[3:11] for n in names]}).to_csv(tmp_path / "labels.csv", index=False)
    # わざと逆順にし、値の無い行も混ぜる
    pd.DataFrame({"filename": list(reversed(names))[:4],
                  "date": [0] * 4,
                  "n_high": [1.0, 2.0, 3.0, 4.0]}).to_csv(tmp_path / "aux.csv", index=False)
    return {"images": images, "labels": tmp_path / "labels.csv",
            "aux": tmp_path / "aux.csv", "names": names}


def test_aux_is_matched_by_filename_not_row_order(workspace):
    """補助の答えは filename で突き合わせること。

    **行番号で並べると、ずれても学習は最後まで走る。**別の日の高気圧の位置を
    答えとして教えることになり、しかも数字はそれらしく出る。
    """
    ds = WeatherMapDataset(workspace["images"], workspace["labels"],
                           aux_csv=workspace["aux"])
    order = {name: i for i, name in enumerate(reversed(workspace["names"]))}
    for i in range(len(ds)):
        name = ds.df["filename"].iloc[i]
        _, _, _, aux = ds[i]
        if order[name] < 4:
            assert aux.item() == pytest.approx(order[name] + 1.0)
        else:
            assert torch.isnan(aux).all(), f"{name} は値が無いはずなのに {aux}"


def test_rows_without_aux_are_kept(workspace):
    """補助の答えが無い行を捨てないこと。

    捨てると画像が減り、10ラベルの学習まで痩せる。NaN のまま返して
    損失の側で飛ばせばよい。
    """
    with_aux = WeatherMapDataset(workspace["images"], workspace["labels"],
                                 aux_csv=workspace["aux"])
    without = WeatherMapDataset(workspace["images"], workspace["labels"])
    assert len(with_aux) == len(without) == 6


def test_aux_stats_use_only_the_given_rows(workspace):
    """標準化の統計は、渡した行だけから求めること。

    全行から求めると、検証・テストの分布が学習に漏れる。
    """
    ds = WeatherMapDataset(workspace["images"], workspace["labels"],
                           aux_csv=workspace["aux"])
    values = ds.aux.numpy().ravel()
    have = [i for i in range(len(ds)) if not np.isnan(values[i])]
    mean, std = compute_aux_stats(ds, have[:2])
    assert mean[0] == pytest.approx(np.mean([values[i] for i in have[:2]]))
    # 全行から求めた値とは違う(漏れていない)
    all_mean, _ = compute_aux_stats(ds, range(len(ds)))
    assert mean[0] != pytest.approx(all_mean[0])


def test_standardisation_is_applied_and_keeps_missing_as_missing(workspace):
    ds = WeatherMapDataset(workspace["images"], workspace["labels"],
                           aux_csv=workspace["aux"])
    ds.set_aux_stats([2.5], [1.0])
    seen = [ds[i][3].item() for i in range(len(ds))]
    assert any(np.isnan(v) for v in seen), "欠測が埋められてしまっている"
    assert min(v for v in seen if not np.isnan(v)) == pytest.approx(1.0 - 2.5)


def test_a_column_with_no_values_does_not_produce_nan_stats(workspace, tmp_path):
    """1件も値の無い**列**があっても、統計がNaNにならないこと。

    NaN の平均を引くと、その列は全行 NaN のままになる ―― それは意図どおり
    (損失から飛ばされる)だが、他の列まで巻き添えにしてはいけない。
    検出が一度も成功しなかった特徴量は実際にありうる。
    """
    pd.DataFrame({"filename": workspace["names"],
                  "never_found": [np.nan] * 6,
                  "n_high": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}).to_csv(
        tmp_path / "partial.csv", index=False)
    ds = WeatherMapDataset(workspace["images"], workspace["labels"],
                           aux_csv=tmp_path / "partial.csv")
    mean, std = compute_aux_stats(ds, range(len(ds)))
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert mean[1] == pytest.approx(3.5)      # 値のある列は正しく求まる

    ds.set_aux_stats(mean, std)
    aux = ds[0][3]
    assert torch.isnan(aux[0]), "値の無い列は NaN のままであるべき"
    assert torch.isfinite(aux[1]), "値のある列まで NaN になっている"


def test_an_aux_file_with_no_matching_rows_is_refused(workspace, tmp_path):
    """1件も突き合わない補助CSVは、黙って進まず止めること。

    そのまま進むと、補助の損失が常に0のまま学習が完走する。数字は出るので
    「効かなかった」と読んでしまう ―― 実際には配線が繋がっていない。
    """
    pd.DataFrame({"filename": ["別の名前.png"], "n_high": [1.0]}).to_csv(
        tmp_path / "wrong.csv", index=False)
    with pytest.raises(ValueError, match="突き合いません"):
        WeatherMapDataset(workspace["images"], workspace["labels"],
                          aux_csv=tmp_path / "wrong.csv")
