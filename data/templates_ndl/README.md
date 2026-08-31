# H / L の文字テンプレート(2000〜2022年の天気図用)

`data/templates/` は**2023年以降**の気象庁PDF版から切り出したもので、
国立国会図書館由来の古い天気図には当たらない。書体も配置もほぼ同じだが、
**記号の線が細い**(塗りの割合が11%、2023年以降は19〜33%)ためで、
大きさの調整では直らない(`--letter-size` を1052〜1900px相当のどこに
置いても検出はほぼ0個だった)。

そこで、その時代の天気図から切り直したものをここに置く。

    H.jpg  H2.jpg  H3.jpg ...      高気圧の「H」
    L.jpg  L2.jpg  L3.jpg ...      低気圧の「L」

* **拡張子はjpgでもpngでもよい。**2値化した結果の食い違いは品質75でも
  0.04〜0.31%しかない(実測)
* 末尾の数字は自動でまとめて数える(H2 も H_b も「H」として扱う)
* 白地に黒でも自動で反転する
* 実測では H 5枚・L 7枚で足りた

## 一番多い失敗

**周りの等圧線を消し忘れること。**線が1本入っているだけで、そのテンプレートは
「自分自身にしか当たらない」ものになる。天気図側は記号だけを探しているのに、
テンプレートには余分な線があるので、一致スコアがどこでも下がるためである。

見た目では気づきにくいので、当てる前に必ず点検すること:

```bash
python -m scripts.check_templates --templates data/templates_ndl

# その時代の天気図に実際に当ててみる(こちらが本番の確認)
python -m scripts.check_templates --templates data/templates_ndl --chart <天気図>
```

## 使い方

```bash
python scripts/predict.py <古い天気図> --annotate --no-preprocess `
    --templates data/templates_ndl --weights weights/model_annot.pt `
    --save-annotated annotated.png
```

**注意**: 検出が直っても、分類の確信度は当てにならない。ラベル付きデータも
同梱の重みもすべて2023年以降の天気図だけで作られているので、古い天気図は
学習時と違う絵になる。
