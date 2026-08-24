"""分類ラベルの定義。データが増えたらここを更新する。"""

LABELS = [
    "winter_pressure_pattern",   # 西高東低（冬型）
    "nankigan_low",              # 南岸低気圧
    "japan_sea_low",             # 日本海低気圧
    "futatsudama_low",           # 二つ玉低気圧
    "typhoon",                   # 台風
    "migratory_high",            # 移動性高気圧（帯状高気圧を統合）
    "pacific_high",              # 太平洋高気圧型
    "front_passage",             # 前線通過（寒冷前線・温暖前線）
    "stationary_front",          # 停滞前線
    "okhotsk_high",              # オホーツク海高気圧
]

LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}
INDEX_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# ラベリングツール・Web UIでの表示用（内部的なキーは英語のまま統一する）
LABEL_JA = {
    "winter_pressure_pattern": "西高東低（冬型）",
    "nankigan_low": "南岸低気圧",
    "japan_sea_low": "日本海低気圧",
    "futatsudama_low": "二つ玉低気圧",
    "typhoon": "台風",
    "migratory_high": "移動性高気圧",
    "pacific_high": "太平洋高気圧型",
    "front_passage": "前線通過（寒冷前線・温暖前線）",
    "stationary_front": "停滞前線",
    "okhotsk_high": "オホーツク海高気圧",
}


# ラベルの併用についての決まり
#
# 1枚の天気図に複数のラベルが付く(2401件のうち1107件が2個以上)。どう併用するかを
# 決めておかないと、同じ気圧配置に違うラベルが付き、モデルは矛盾した答えを要求される。
#
#   futatsudama_low(二つ玉低気圧)は、japan_sea_low と nankigan_low を「置き換える」。
#   個別の低気圧ラベルは付けない。
#
# 106件のうち56件がこのラベル単独で、両方を伴う例は0件だった。一方で片方だけを伴う
# 例が9件あり、そのうち6件が2026年、8件が2025年3月以降に集中していた -- 途中で基準が
# 変わっていたということ。9件は scripts/apply_label_rule.py で規約に合わせた。
#
# 逆向きの例(japan_sea_low と nankigan_low が揃っていて futatsudama_low が無い)が
# 13件残っている。これは規約の問題ではなく「本当に二つ玉か」の判定なので、
# 天気図を見て決める必要がある。
#
# ここに書かれていない組み合わせについては、まだ決まりが無い。前線(front_passage /
# stationary_front)は低気圧ラベルと自由に併用されており、それは物理的に自然である。
#
# typhoon(台風)は、人が天気図を見て付けるのをやめ、機械的に付与している。
#
#   気象庁ベストトラックで階級がTS以上(grade 3〜5)の台風の中心が
#     北緯20〜46度・東経123〜154度
#   にある時刻に付ける。本土・南西諸島・小笠原・南鳥島に加え、南から接近中の
#   範囲を含む。付与は scripts/typhoon_from_besttrack.py。
#
# 人手でやめた理由: 同じ2024年9月を1年以上あけて判定し直したら、12日が30日になった
# (κ 0.400)。台風は天気図に明記されるので見落としではなく、「どこまでを日本への影響と
# みなすか」が書かれていなかったための揺れ。実際、2023年はフィリピン沖の台風も
# 「あり」としていた(28件)一方、2024年以降は付け忘れが56件あった。
#
# ベストトラックは中心位置と強度の数値記録で、ERA5とも天気図とも独立している。
# だから天気図モデルにも格子モデルにも不当な有利を与えない -- ここは重要で、
# もしERA5の気圧場から規則でラベルを作ると、格子モデルは自分の入力の関数を
# 再現するだけになり、入力どうしの比較が成立しなくなる。
#
# 機械ラベルは、人手で付けた過去のラベルとκ 0.801で一致する。
# **人間が自分自身と一致する度合い(κ 0.400)より高い。**
# 1951年以降が公開されているので、2000〜2022年の天気図にもそのまま展開できる。


# 決まりを変えたときは、ここを書き換えてから scripts/apply_label_rule.py で
# 既存のCSVに当てること。CSVだけ直すと、次にラベルを付ける人が同じ揺れを起こす。


def parse_labels(label_field: str) -> list:
    """"winter_pressure_pattern|japan_sea_low" のようなパイプ区切りを分解する。

    ラベルCSVを読むだけの処理(集計・突き合わせ)からも使うため、
    torchに依存しないこのモジュールに置いている。
    """
    return [l for l in str(label_field).split("|") if l in LABEL_TO_INDEX]

