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
