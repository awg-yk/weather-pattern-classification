"""検出の設定が1つだけであることを固定するテスト。

以前は2023年を境に設定を打ち分けていた(古い天気図では
letter_size="auto"・detect_threshold=0.55)。**前処理がすべての天気図を
1453x1500 に揃えるようにしてから、打ち分けは不要になった。**

実測(2000-01-01の天気図、しきい値0.65・テンプレート原寸):
    揃える前 1499x1548  H 0 / L 0
    揃えた後 1453x1500  H 3 / L 4   <- 2023年以降の平均 H 2.8 / L 3.9〜4.2 と同水準

**この値は runs/cv_annot_boxes を作ったときのもので、同梱の重みはこの設定で
描いた画像で学習してある。**変えると、モデルに学習時と違う絵を渡すことになる。
"""

import inspect

from src import quicklook
from src.quicklook import DETECTION, classify_and_show, make_annotated


def test_there_is_one_setting_not_one_per_era():
    assert DETECTION == {"letter_size": 1.0, "detect_threshold": 0.65}


def test_the_era_split_is_gone():
    """時代で打ち分ける関数が復活していないこと。"""
    source = inspect.getsource(quicklook)
    for gone in ("detection_settings", "TRAINING_ERA", "OLD_ERA"):
        assert gone not in source, f"時代の打ち分けが戻っている: {gone}"


def test_the_defaults_can_still_be_overridden():
    """明示すれば、そちらが使われること。"""
    for func in (classify_and_show, make_annotated):
        params = inspect.signature(func).parameters
        assert params["letter_size"].default is None
        assert params["detect_threshold"].default is None


def test_the_canonical_size_and_the_threshold_stay_together():
    """揃える大きさを変えると、記号の大きさが変わってしきい値も合わなくなる。
    片方だけ動かせないことを、ここで結びつけておく。"""
    from scripts.preprocess_jma import CANONICAL_SIZE

    assert CANONICAL_SIZE == (1453, 1500)
    assert DETECTION["letter_size"] == 1.0
