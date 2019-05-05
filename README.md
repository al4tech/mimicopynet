mimicopynet
====

mimicopynetは耳コピ(music transcription)を自動的に行うことを目的とした，pythonパッケージです．
chainerで実装されています．

現在は，音源をスペクトル分解し，周波数×時間の画像を用意したあと，その画像から，それぞれの時間に置いて，任意のピッチの音がなっているかを判定するCNN(BasicCNN)を用意しています．

## Requirement
### Python環境
python3で実装されています。

以下必要なパッケージ(anacondaでデフォルトで入っているものは除く)

- chainer
- pretty_midi
- librosa

pretty_midiは以下のところからクローンして、インストールしてください。
https://github.com/craffel/pretty-midi  

### データセット
現在，データとしてMusicNetを使うことを前提としています．
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
wavescoredata には，波形のデータと譜面のデータが，簡素なフォーマットで格納されています．

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
```python
model.load_cqt_inout("testdata.npz")
model.learn()
```

学習されたモデルは，resultディレクトリに保存されます．
学習済みモデルをロードして，耳コピを行います．
```python
model.load_model("result/model_50000.npz")
model.transcript("test.wav", "test.mid")
```

CQTの実部と虚部を使うコードは以下の通りです．
```python
mcn.data.make_cqt_inout("wsdata", "data_rawmode.npz", mode='raw')

model = mcn.model.BasicCNN(input_cnl=2)

model.load_cqt_inout("data_rawmode.npz")
model.learn()

#model.load_model("result/model_50000.npz")
#model.transcript("test.wav", "test.mid", mode='raw')
```

## Future Work
まだまだ，mimicopynetは改善していきます．
例えば，今はある音程のピアノの音がなっているかの２値判別の問題を解いていますが，音がなり始めかどうかの２値判別の方が良いかもしれません．

今後キリの良いところで英語化をしていく予定です．

## Contribution

marshi(Yoshikawa Masashi)  
yos1up(Yoshida Yuki)

コードの分量はmarshiが多いと思いますが，yos1upには色々議論したりして，多大な貢献をしていただきました．そもそも，最初にディープラーニング耳コピを始めたのはyos1upでした．

## Author

marshi(Yoshikawa Masashi)  
yos1up(Yoshida Yuki)
