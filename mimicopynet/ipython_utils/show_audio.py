import numpy as np
import IPython.display

def show_audio(raw_waveform, sr, repeat=1):
    '''
    ipythonで波形再生するフォームを表示します．
        raw_waveform <np.ndarray (sample_num, channel_num) np.int16>: 各値は -32768〜+32767
        sr <scalar> : サンプリング周波数
        repeat <int >= 1> : ループ回数
    '''
    return IPython.display.Audio(list(np.tile(raw_waveform,[repeat,1]).T), rate=sr)

