mimicopynet
====

mimicpynetはピアノ演奏の耳コピモデルをベンチマークするための実装です。chainerで実装されています。
まだ開発途中で、今後コメントやPEP8に準拠した書き方にしていきます。

## Requirement
python3で実装されています。

以下必要なパッケージ(anacondaでデフォルトで入っているものは除く)
・chainer
・pretty_midi
・midi2audio
その他必要なもの
・FluidSynth

chainer,midi2audioはpipでインストールできます。

pretty_midiは以下のところからクローンして、インストールしてください。
https://github.com/craffel/pretty-midi

Fluidsynthのインストールはここを参考してください
http://kujirahand.com/blog/index.php?Mac%20OS%20X%E3%81%A7MIDI%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%82%92%E3%82%AA%E3%83%BC%E3%83%87%E3%82%A3%E3%82%AA%E3%81%AB%E5%A4%89%E6%8F%9B%E3%81%99%E3%82%8B%E6%96%B9%E6%B3%95

## Usage

インポート

```python
import mimicopynet as mcn
```

musicnetからnpz形式で，訓練データを作成する

```python
mcn.data.piano_train_data("musicnet.npz", "musicnet_metadata.csv", "musicnet_data")
```

データ読み込み

```python
data = np.load("musicnet_data/1733.npz")
```
