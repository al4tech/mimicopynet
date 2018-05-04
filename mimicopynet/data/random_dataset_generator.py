import numpy as np
import pickle
from .make_random_song import make_random_song
from .midi_to_wav import midi_to_wav
from .preprocess import make_cqt_input
from .midi_to_score import midi_to_score

from chainer import datasets, cuda
import chainer


# class dataset_generator():
#     pass

class RandomDataset(chainer.dataset.DatasetMixin): # chainer.dataset.DatasetMixin を継承する必要はあるか？
    def __init__(self, num_samples, gpu=None):
        self.num_samples = num_samples
        self.gpu = gpu
        if gpu is not None:
            cuda.get_device(gpu).use()
        self.refresh()
    def generateTupleDataset(self, num_samples, inst=0):
        stride = 4
        length = num_samples * stride + 128 # 512/44100秒 の個数
        make_random_song('_.mid', len_t=length*512/44100, inst=inst)
        midi_to_wav('_.mid','_.wav')
        cqt = make_cqt_input('_.wav', mode='raw', scale_mode='midi')
        score = midi_to_score('_.mid')
        if cqt.shape[-1] < length:
            cqt = np.concatenate([cqt, np.zeros([cqt.shape[0], cqt.shape[1], length - cqt.shape[-1]], dtype=np.float)], axis=2)
        elif cqt.shape[-1] > length:
            cqt = cqt[:,:,length]
        if score.shape[-1] < length:
            score = np.c_[score, np.zeros([score.shape[0], length - score.shape[-1]], dtype=np.int)]
        elif score.shape[-1] > length:
            score = score[:,length]
        assert(cqt.shape[-1]==score.shape[-1]==length)
        # cqt:   (ch, scale, length) float
        # score: (scale, length) int
        cqt = cqt.astype(np.float32)
        score = (score > 0).astype(np.int32) # 0-128を 0-1に変換している
        if self.gpu is not None:
            cqt = cuda.to_gpu(cqt)
            score = cuda.to_gpu(score)
        spect_list = [cqt[:,:,i*stride:i*stride+128] for i in range(num_samples)]
        score_list = [score[:,i*stride:i*stride+128] for i in range(num_samples)]
        return datasets.TupleDataset(spect_list, score_list)
    def refresh(self, num_samples=None):
        if num_samples is None: num_samples = self.num_samples
        self.dataset = self.generateTupleDataset(num_samples)
    def __len__(self):
        return len(self.dataset)
    def get_example(self, i):
        return self.dataset[i]


        '''
        ちょっと計算：
        1エポックずつデータを返す仕様にしてみよう．
        1エポックのデータ量は
        例えば 1万件の (2,128,128)float32 (=128KB) と (128,128)int32 (=64KB) は約 2GB (スライスの共通部分も多いので実際はそれより小さい)
        なお，だいたい1サンプルは 1秒分のウィンドウ．
        midi→wav変換は，1000秒分変換するのに約9秒かかる．
        wav→cqt変換(midi scale)は，1000秒分に約23秒．

        例えば 0.01秒ずつずらしてサンプルとして使う場合は，
            1万サンプル(100秒分)を産むのに，約3.2秒かかる．
        一方で，gpu使ってミニバッチ学習してみると，1万サンプルが約3秒(バッチサイズ1000の場合)．同じくらいなので許容範囲か．
        '''



        



