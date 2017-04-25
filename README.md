mimicopynet
====

mimicopynetは耳コピ(music transcription)を自動的に行うことを目的とした，pythonパッケージです．
chainerで実装されています．


## Requirement
python3で実装されています。

以下必要なパッケージ(anacondaでデフォルトで入っているものは除く)

- chainer
- pretty_midi
- librosa

pretty_midiは以下のところからクローンして、インストールしてください。
https://github.com/craffel/pretty-midi

現在，MusicNetを使うことを前提としています．
MusicNetはMIRで研究されることを目的とした，音楽のデータセットです．
https://homes.cs.washington.edu/~thickstn/musicnet.html

それぞれ以下のリンクからダウンロードしてください．
- メタデータ(https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv)
- npzファイル(https://homes.cs.washington.edu/~thickstn/media/musicnet.npz)

## Usage

インポート

```python
import mimicopynet as mcn
```

musicnetからピアノのソロ曲のみを，wavescoredataというmimicopynetのデータ形式に変換します．

```python
mcn.data.musicnet_to_wsdata("musicnet.npz", "musicnet_metadata.csv", "wsdata", "Solo Piano") #3つめの引数は，wavescoredataが保存されるディレクトリ
```

wavescoredataから，CQT(Constant Q Transform)を行い，訓練データに整形します．
modeは'abs'と'raw'の２種類があります．
```python
mcn.data.make_cqt_inout("wsdata","testdata.npz", mode='abs')
```

CNNモデルをインスタンス化します．
```python
model = mcn.model.BasicCNN(input_cnl=1)
```

訓練データをロードして，学習させます．
```
model.load_cqt_inout("testdata.npz")
model.learn()
```

学習されたモデルは，resultディレクトリに保存されます．
学習済みモデルをロードして，耳コピを行います．
```
model.load_model("result/model_50000.npz")
model.transcript("test.wav", "test.mid")
```

CQTの実部と虚部を使うコードは以下の通りです．
```
mcn.data.make_cqt_inout("wsdata", "data_rawmode.npz", mode='raw')

model = mcn.model.BasicCNN(input_cnl=2)

model.load_cqt_inout("data_rawmode.npz")
model.learn()

#model.load_model("result/model_50000.npz")
#model.transcript("Niyodo - piano.wav", "test.mid", mode='raw')
```
## Contribution

marshi

yos1up

## Author

marshi

yos1up
