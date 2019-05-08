
import h5py
import numpy as np

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
        self.wave_sr = _AccessorToH5(filename, 'wave_sr')
        self.score = _AccessorToH5(filename, 'score')
        self.score_onset = _AccessorToH5(filename, 'score_onset')
        self.score_sample = _AccessorToH5(filename, 'score_sample')

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
                shape == (パート数, サンプル)
            score:
                shape == (パート数, ノートナンバー, 時刻)
            score_onset:
                shape == (パート数, ノートナンバー, 時刻)

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
        with h5py.File(self.filename, 'r') as f:
            wave_shape = f['wave'].shape
            score_shape = f['score'].shape

            sample_pitch = f['score_sample'][1] - f['score_sample'][0]
            score_index_width = int(round(length_sec * 44100 / sample_pitch))

            if score_index_width < score_shape[-1]:
                score_index_start = np.random.randint(score_shape[-1] - score_index_width + 1)
                score_index_end = score_index_start + score_index_width
            else:
                score_index_start = 0
                score_index_end = score_shape[-1]

            sample_start = f['score_sample'][score_index_start] - sample_pitch//2
            sample_end = f['score_sample'][score_index_end - 1] + sample_pitch//2


            if no_drum:
                possible_parts = np.flatnonzero(f['inst_is_drum'][:] == False)
            else:
                possible_parts = np.arange(score_shape[0])

            if num_part is None:
                parts = possible_parts
            else:
                num_part = min(num_part, len(possible_parts))
                parts = np.sort(np.random.choice(possible_parts, num_part, replace=False))

            wave = f['wave'][parts, sample_start:sample_end]
            score = f['score'][parts, :, score_index_start:score_index_end]
            score_onset = f['score_onset'][parts, :, score_index_start:score_index_end]

        return wave, score, score_onset










