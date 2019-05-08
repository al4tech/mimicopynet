import numpy as np
import pickle
from .make_random_song import make_random_song
from .midi_to_wav import FluidSynth #, midi_to_wav
from .preprocess import make_cqt_input
from .midi_to_score import midi_to_score

from chainer import datasets, cuda
import chainer


# class dataset_generator():
#     pass

class RandomDataset(chainer.dataset.DatasetMixin): # chainer.dataset.DatasetMixin を継承する必要はあるか？
    def __init__(self, num_samples, sound_font, inst=0, gpu=None):
        """
        Args:
            num_samples (int):
                データ数．
            sound_font (str):
                wave を作るためのサウンドフォントファイル (.sf2) へのパス．
            inst:

            gpu:
        """
        self.num_samples = num_samples
        self.sound_font = sound_font
        self.inst = inst
        self.gpu = gpu
        self.level = 0
        if gpu is not None:
            cuda.get_device(gpu).use()
        self.refresh()
    def generateTupleDataset(self, num_samples, **kwargs):
        stride = 16
        length = num_samples * stride + 128 # 512/44100秒 の個数
        make_random_song('_.mid', len_t=length*512/44100, **kwargs)
        fls = FluidSynth(sound_font=self.sound_font)
        fls.midi_to_audio('_.mid','_.wav')
        cqt = make_cqt_input('_.wav', mode='raw', scale_mode='midi')
        score = midi_to_score('_.mid', sampling_rate=44100/512, mode='hold')
        score = np.max(score, axis=0) # .max により，全パートをマージする．

        if cqt.shape[-1] < length:
            cqt = np.concatenate([cqt, np.zeros([cqt.shape[0], cqt.shape[1], length - cqt.shape[-1]], dtype=np.float)], axis=2)
        elif cqt.shape[-1] > length:
            cqt = cqt[:,:,:length]
        if score.shape[-1] < length:
            score = np.c_[score, np.zeros([score.shape[0], length - score.shape[-1]], dtype=np.int)]
        elif score.shape[-1] > length:
            score = score[:,:length]
        assert(cqt.shape[-1]==score.shape[-1]==length)
        # cqt:   (ch, scale, length) float
        # score: (scale, length) int
        cqt = cqt.astype(np.float32)
        score = (score > 0).astype(np.int32) # 連続値から 0 or 1 の二値に変換していることに注意！
        if self.gpu is not None:
            cqt = cuda.to_gpu(cqt)
            score = cuda.to_gpu(score)
        spect_list = [cqt[:,:,i*stride:i*stride+128] for i in range(num_samples)]
        score_list = [score[:,i*stride:i*stride+128] for i in range(num_samples)]
        return datasets.TupleDataset(spect_list, score_list)
    def refresh(self):
        # self.level に応じて，self.inst から inst_set を計算する．
        # レベルが上がるほど高確率で色々な音色を含むようにする？
        self.dataset = self.generateTupleDataset(self.num_samples, inst=self.inst, lmb_start=300.0/(100+self.level))
        self.level += 1
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



        



