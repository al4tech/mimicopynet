import numpy as np
import IPython.display

def show_audio(raw_waveform, sr, repeat=1):
    '''
    ipythonで波形再生するフォームを表示します．
        raw_waveform:
            再生したい波形．以下のいずれかのフォーマット
            # <np.ndarray (sample_num, channel_num) np.int16>: 各値は -32768〜+32767
            # <np.ndarray (sample_num, channel_num) np.float32>: 各値は -1〜+1
            numpy.ndarray (sample_num, channel_num) であればスケールは任意っぽい．
            
        sr <scalar> : サンプリング周波数
        repeat <int >= 1> : ループ回数
    '''
    assert(raw_waveform.ndim == 2)
    """
    if raw_waveform.dtype == np.int16:
        pass
    elif raw_waveform.dtype == np.float32 and np.max(np.abs(raw_waveform)) <= 1:
        raw_waveform = (raw_waveform * 32767).astype(np.int16)
    else:
        raise ValueError
    """

    return IPython.display.Audio(list(np.tile(raw_waveform,[repeat,1]).T), rate=sr)

