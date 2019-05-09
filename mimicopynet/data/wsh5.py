
import h5py
import numpy as np
from chainer import cuda, dataset, datasets

def get_array_in_fixed_size(src, axis, size_dest, start_src):
    """
    src の axis 方向から size_dest の長さだけ取り出した array を返します．
    端はゼロパディングされ，必ず（その方向に）size_dest の長さの配列が返ります．
    --------
    Args:
        src (numpy.ndarray or .h5 handle):
            
        axis (int):
            
        size_dest (int):

        start_src (int):

    Returns:
        dest (numpy.ndarray):

    """
    shape_src, dtype_src = src.shape, src.dtype

    size_src = shape_src[axis]
    dest = np.zeros([size_dest if i==axis else n for i, n in enumerate(shape_src)], dtype=dtype_src)
    # if start_src is None:
    #     start_src = np.random.randint(size_src + size_dest - 1) - size_dest + 1
    end_src = start_src + size_dest
    start_dest = 0
    end_dest = size_dest

    if start_src >= size_src:
        return dest
    elif start_src < 0:
        start_dest -= start_src
        start_src = 0

    if end_src <= 0:
        return dest
    elif end_src > size_src:
        end_dest -= (end_src - size_src)
        end_src = size_src

    idx_src = tuple(
        slice(start_src, end_src) if i==axis else slice(None)
        for i in range(len(shape_src))
    )
    idx_dest = tuple(
        slice(start_dest, end_dest) if i==axis else slice(None)
        for i in range(len(shape_src))
    )
    dest.__setitem__(idx_dest, src.__getitem__(idx_src))
    return dest

class _AccessorToH5(object):
    """
    __init__ 以外の任意のメソッドやアクセサ呼び出しで，ファイルが一瞬開かれるので注意！
    """
    def __init__(self, filename, property_name):
        self._filename = filename
        self._property_name = property_name
    def __getitem__(self, item): # access by []
        with h5py.File(self._filename, 'r') as f:
            ret = f[self._property_name].__getitem__(item)
        return ret
    @property
    def shape(self):
        with h5py.File(self._filename, 'r') as f:
            ret = f[self._property_name].shape
        return ret
    @property
    def dtype(self):
        with h5py.File(self._filename, 'r') as f:
            ret = f[self._property_name].dtype
        return ret


    
class WSH5(object):
    """
    wsh5 形式のファイルから効率的に読みだすためのクラス．

    [使用例]
        ws = WSH5('hoge.h5') # この時点ではファイルを開いていない．

        # 以下の data は numpy.ndarray
        data = ws.wave[0, 3000000:3000000+512] # ファイルから O(512) の時間で読み出される（ファイルは一瞬だけ開かれる）
        data = ws.score[0, 60:72, 1000:1100] # 同様
        data = ws.score[:] # 全体が欲しい場合は [:] で numpy.ndarray になる．（重い）
        sh = ws.score.shape # shape はこれで取れる．

        少ない記述量で読み出せるが，ストレージへのアクセスしすぎに注意！

    Methods:
        sample: データをサンプリングする．便利．

    Property:
        filename (str)

    """
    def __init__(self, filename):
        self.load_wsh5(filename)

    def load_wsh5(self, filename):
        self.filename = filename
        self.wave = _AccessorToH5(filename, 'wave')
        # self.wave_sr = _AccessorToH5(filename, 'wave_sr')
        self.score = _AccessorToH5(filename, 'score')
        self.score_onset = _AccessorToH5(filename, 'score_onset')
        self.score_sample = _AccessorToH5(filename, 'score_sample')

        # データの healthy check をいくつか行う．
        # score_sample が等差数列になっていることとか，wave の総サンプル数からはみ出していないこととか．
        with h5py.File(self.filename, 'r') as f:
            score_sample = f['score_sample'][:]
            wave_shape = f['wave'].shape
            self.wave_sr = f['wave_sr'][:]
            score_shape = f['score'].shape

        intervals = score_sample[1:] - score_sample[:-1]
        assert all(intervals == intervals[0]), "`score_sample` should be arithmetic sequence."
        self.score_sample_interval = interval[0]
        self.score_sample_origin = score_sample[0]
        assert 0 <= score_sample[0] and score_sample[-1] < wave_shape[-1], "`score_sample` has out-of-range values (compared to `wave`)"

        assert self.wave_sr == 44100
        assert self.score_sample_interval == 512
        assert len(wave_shape) == 2
        assert len(score_shape) == 3

    def to_sample_of_wave(self, sample_of_score):
        """
        与えられた score のサンプル番号と同時刻に相当する wave のサンプル番号を返します
        score_sample が等差数列であることを仮定しています！
        """
        return self.score_sample_origin + self.score_sample_interval * sample_of_score

    def to_sample_of_score(self, sample_of_wave):
        """
        与えられた wave のサンプル番号と同時刻に相当する score のサンプル番号を返します
        score_sample が等差数列であることを仮定しています！
        """        
        return int(floor((sample_of_wave - self.score_sample_origin) / self.score_sample_interval))

    @staticmethod
    def save_wsh5(filename, **kwargs):
        with h5py.File(filename, 'w') as f:
            for k, v in kwargs.items():
                f.create_dataset(k, data=v)

    def sample(self, length_sec, no_drum=False, num_part=None):
        """
        指定された秒数のデータをランダムにサンプリングして返す．
        ------
        Args:
            length_sec (float):
                欲しい秒数．長すぎる場合は曲全体が返る．
            no_drum (bool):
                これを指定した場合は，ドラムパートは返らなくなります．
            num_part (int or None):
                指定されたパート数だけランダムに選んで返します．
                None または多すぎる場合は，返りうる全パートを返します．

        Returns:
            wave:
                shape == (パート数, samples_of_wave)
            score:
                shape == (パート数, ノートナンバー, samples_of_score)
            score_onset:
                shape == (パート数, ノートナンバー, samples_of_score)

        Examples:
            ```
            import numpy as np
            import matplotlib.pyplot as plt
            import mimicopynet as mcn
            ws = mcn.data.WSH5('_hoge.h5')
            wave, score, onset = ws.sample(3.0, num_part=2, no_drum=True)

            # listen to sound
            mcn.ipython_utils.show_audio(np.mean(wave, axis=0).reshape(-1, 1), 44100)

            # visualize onset
            plt.imshow(np.sum(onset, axis=0).astype(np.bool), origin='lower')
            plt.show()
            ```
        """
        wave_len, score_len = self.get_sample_length(length_sec)
        with h5py.File(self.filename, 'r') as f:
            wave_shape = f['wave'].shape
            score_shape = f['score'].shape

            """
            sample_interval = f['score_sample'][1] - f['score_sample'][0]
            score_index_width = int(round(length_sec * 44100 / sample_interval))

            if score_index_width < score_shape[-1]:
                score_index_start = np.random.randint(score_shape[-1] - score_index_width + 1)
                score_index_end = score_index_start + score_index_width
            else: # length_sec が（スコア情報に対して）長すぎるケース
                score_index_start = 0
                score_index_end = score_shape[-1]

            sample_start = f['score_sample'][score_index_start]
            sample_end = f['score_sample'][score_index_end - 1] + sample_interval
            # NOTE この wave のサンプル数の算出方法は，
            # midi_to_score の仕様（インデックス i に 時刻 fs*i - fs*(i+1) の情報が格納されている）
            # に影響を受けています
            """
            wave_start = np.random.randint(1 - wave_len, wave_shape[-1])
            wave = get_array_in_fixed_size(f['wave'], axis=1, size_dest=wave_len, start_src=wave_start)
            score_start = self.to_sample_of_score(wave_start)
            score = get_array_in_fixed_size(f['score'], axis=2, size_dest=score_len, start_src=score_start)
            score_onset = get_array_in_fixed_size(f['score_onset'], axis=2, size_dest=score_len, start_src=score_start)

        if no_drum:
            possible_parts = np.flatnonzero(f['inst_is_drum'][:] == False)
        else:
            possible_parts = np.arange(score_shape[0])

        if num_part is None:
            parts = possible_parts
        else:
            num_part = min(num_part, len(possible_parts))
            parts = np.sort(np.random.choice(possible_parts, num_part, replace=False))

        return wave, score[parts], score_onset[parts]

        def get_sample_length(length_sec):
            """
            上の sample が返してくる値たちの 時間軸の size のみをあらかじめ計算するメソッド．
            sample もこれを使う
            """
            samples_of_wave = int(round(length_sec * self.wave_sr))
            samples_of_score = int(round(length_sec * self.wave_sr / self.score_sample_interval))
            return samples_of_wave, samples_of_score




class WSH5Dataset(dataset.DatasetMixin):
    def __init__(self, wsh5s, size, length_sec=1.0, no_drum=True, num_part=3, gpu=-1):
        """
        注意：このデータセットには「レベル」(self.level)という概念があります．
        初期化時はレベルは0です．1エポック終了ごとに「レベル」は勝手に1ずつ上昇していきます．
        レベルが高いほど，難しいデータがサンプルされるように，調節することもできます．（以下参照）

        ↑これはまだ実装されていません．（エポック終了時の処理とかをこちら側では指定できない・・・）
        TODO どちらかというと，Dataset ではなく Iterator を作った方が良さそう？？？
        --------
        Args:
            wsh5s (iterable of str):
                データのランダム抽出元となる，
                WSH5形式のファイルへのパスを表す文字列の iterable
            size (int):
                データセットのサイズ．これは「1エポック」の件数を決めるためだけに用いられる．
            length_sec (float or callable):
                1つのデータに含まれる秒数．定数で与えるか，レベル(int)を引数に float 値を返す callable を指定．
            no_drum (bool or callable):
                ドラムパートを禁止するか否か．定数で与えるか，レベル(int)を引数に bool 値を返す callable を指定．
            num_part (int or callable):
                抽出するパート数．定数で与えるか，レベル(int)を引数に int 値を返す callable を指定．

            gpu (int):

        """
        self.wsh5s = wsh5s
        self.size = size
        self.length_sec = length_sec
        self.num_part = num_part
        self.level = 0
        self.gpu = gpu
        self.xp = np if gpu==-1 else cuda.cupy
        super(self, WSH5Dataset).__init__()

    def __len__(self):
        return self.size

    def get_example(self, i):
        """
        ランダムにデータを返す．
        ----
        Args:
            i (int): インデクス（使われない）
        Returns:
            以下の tuple
                wave (xp.ndarray):
                    波形情報．shape==(samples_of_wave,) dtype==xp.float32
                    正規化はされていないので注意！
                score_feats (xp.ndarray):
                    スコア情報．shape==(2, 128, samples) dtype==xp.int32
                    score_feats[0] は hold 譜面．(ノートオンからオフまで 1 が入る．それ以外は 0)
                    score_feats[1] は onset 譜面．(ノートオンの瞬間だけ 1 が入る．それ以外は 0)

            （タスク設計は，loss関数の側で行ってください．）

        """
        ws = WSH5(np.random.choice(self.wsh5s))
        length_sec = self.length_sec(self.level) if callable(self.length_sec) else self.length_sec
        no_drum = self.no_drum(self.level) if callable(self.no_drum) else self.no_drum
        num_part = self.num_part(self.level) if callable(self.num_part) else self.num_part
        wave, score, score_onset = ws.sample(length_sec, no_drum, num_part)
        # TODO
        # 混ぜ合わせを 1:1 からずらしたり，逆位相にしたり，オフセット少しずらしたり，
        # ホワイトノイズなど加えたり，ピッチシフトしたり，など色々なオーグメンテーションができる．

        # パートのマージ．
        wave = np.sum(wave, axis=0) # (samples_of_wave,)
        score = np.sum(score, axis=0) # (pitch, samples_of_score)
        score_onset = np.max(score_onset, axis=0) # (pitch, samples_of_score)

        score_feats = np.stack(score, score_onset) # (2, pitch, samples_of_score)

        # score のバイナリ化．
        score_feats = score_feats.astype(np.bool).astype(np.int32)

        if self.gpu != -1:
            wave = cuda.to_gpu(wave)
            score_feats = cuda.to_gpu(score_feats)
        # TODO GPUに転送するタイミングはもっと遅い方が効率的かも（例えばバッチを組んでからとか？）

        return wave, score_feats






