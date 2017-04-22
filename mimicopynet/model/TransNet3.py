# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:42:53 2016

@author: marshi, yos1up

TransNet.pyよりコピー。好き勝手に編集することにする。 (17/3/4)
"""

import numpy as np
import chainer
from chainer import optimizers, cuda
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import scipy as sp

def tocpu(x):
    if not isinstance(x, np.ndarray):
        return cuda.to_cpu(x)
    return x
        
"""
def mulaw(wav, mu=255):
    wav = np.sign(wav)*np.log(1+mu*np.abs(wav))/np.log(1+mu)
    # wav = np.round((wav + 1) / 2 * mu).astype(np.int32)  
    wav = (wav + 1.) / 2 * mu
    return wav  
"""
def quantize_linear(lot):
    # -1〜1のnp.float32を0〜255のnp.int32に
    return [(((tup[0]+1.)*128).astype(np.int32), tup[1]) for tup in lot]
"""
def quantize_mulaw(lot):
    # -1〜1のnp.float32を0〜255のnp.int32に
    return [(mulaw(tup[0]), tup[1]) for tup in lot]
"""

def get_filters(nns, siz=4096, cqt=False, fs=44100):
    ret = []
    for nn in nns:
        f = 441. * 2.**((nn-69)/12)
        ret.append(np.exp(1j*(np.arange(siz)-siz/2)/fs*f*2*np.pi) * np.hanning(siz))
    return ret

def get_spectrum(X, s, filters): # ピッチのスペクトラム
    '''
    X: 波形、s: 中央のサンプル、filters: フィルターのリスト
    '''
    ret = []
    for i,fi in enumerate(filters):
        ret.append(np.sum(X[s-len(fi)//2:s-len(fi)//2+len(fi)] * fi, axis=-1))
    return np.array(ret) # Xが1-dimなら1-dim (#filter), 2-dim(#batch, #sample)なら2-dimになる (#filter, #batch)  中身は複素数なことに注意



class yosnet_ft(chainer.Chain):
    '''
    フォワードモデルだけ
    '''
    def __init__(self, a_nn=list(np.arange(128)), slen=5120, ssize=1):
        self.a_nn = a_nn
        self.a_dim = len(a_nn)
        self.slen = slen # 5120 一つのセグメントの長さ（wavのサンプル数）
        self.ssize = ssize # 1 一つの入力あたりのセグメントの個数
        self.filters = get_filters(list(np.arange(21, 109)), siz=slen)  
        super(yosnet_ft, self).__init__(
            l1 = L.Linear(None, 128),
            b1 = L.BatchNormalization(128),
            l2 = L.Linear(None, 128),
            b2 = L.BatchNormalization(128),
            l3 = L.Linear(None, 128),
            b3 = L.BatchNormalization(128),
            l4 = L.Linear(None, 128),
            b4 = L.BatchNormalization(128),
            l5 = L.Linear(None, 128),
            b5 = L.BatchNormalization(128),                                   
            l6 = L.Linear(None, self.a_dim)
        )
    def preprocess(self, lot): # 入力データの前処理をこいつで規定する
        ret = []
        for x,y in lot:
            cs = get_spectrum(x, len(x)//2, self.filters)
            ret.append((np.array([c.real for c in cs] + [c.imag for c in cs], dtype=np.float32), y))
        return ret
    def __call__(self, x, mode='train'):
        '''
        x <Variable (bs, #filters*2) float32>
        returns <Variable (bs, 128(=self.a_dim), ssize) float32> 0〜1で出して。
        '''
        testflg = (mode=='test')
        bs = x.shape[0]
        y = x
        for i in range(1,6):
            y = self['b'+str(i)](F.relu(self['l'+str(i)](y)),testflg)
        y = F.sigmoid(self.l6(y)) # (bs, ssize*128)
        y = F.transpose(F.reshape(y, (bs, self.ssize, self.a_dim)), (0, 2, 1))
        return y


class yosnet(chainer.Chain):
    '''
    フォワードモデルだけ
    '''
    def __init__(self, a_dim=128, slen=5120, ssize=1):
        self.a_dim = a_dim
        self.slen = slen # 5120 一つのセグメントの長さ（wavのサンプル数）
        self.ssize = ssize # 1 一つの入力あたりのセグメントの個数   
        super(yosnet, self).__init__(
            l1 = L.Linear(None, 1280),
            b1 = L.BatchNormalization(1280),
            l2 = L.Linear(None, 320),
            b2 = L.BatchNormalization(320),
            l3 = L.Linear(None, self.a_dim)
        )
    def preprocess(self, lot): # 入力データの前処理をこいつで規定する
        return quantize_linear(lot)
    def __call__(self, x, mode='train'):
        '''
        x <Variable (bs, ssize*512) int32>
        returns <Variable (bs, 128(=self.a_dim), ssize) float32> 0〜1で出して。
        '''
        testflg = (mode=='test')
        assert(x.shape[1] == self.ssize * self.slen)
        bs = x.shape[0]
        xp = cuda.get_array_module(x.data)
        h = chainer.Variable(x.data.astype(xp.float32)/128-1)
        h = F.reshape(h, (bs*self.ssize, self.slen))
        y = F.sigmoid(self.l3(self.b2(F.relu(self.l2(self.b1(F.relu(self.l1(h)),testflg))),testflg))) # (bs*ssize, 128)
        y = F.transpose(F.reshape(y, (bs, self.ssize, self.a_dim)), (0, 2, 1))
        return y




class wavenet(chainer.Chain):
    '''
    耳コピ用のWavenetです．（フォワードモデルだけ）
    音声データはmu=255のmu-lawエンコードした後のデータをonehotベクトルで入力します
    出力データは128個の音階の0～1データです．
    '''
    def __init__(self, embed_dim=16, a_dim=128):
        # embed_dimに1を指定するとembed_idしなくなります（波形の連続値がそのまま放り込まれるようになる）。
        self.mu = 255#μ-rawのμです
        self.train_sample = 2**9#512サンプルの1出力です
        self.total_iter = 0
        self.loss = None
        self.a_dim = a_dim # 出力の次元
        self.slen = 5120 # 定義したけどまだ使ってない
        self.ssize = 1 # 定義したけどまだ使ってない
        self.embed_dim = embed_dim
        self.lossfrac = np.zeros(2) # loss平均計算用
        self.acctable = np.zeros((2, 2)) # 正解率平均計算用
        super(wavenet, self).__init__(
            last1 = L.Convolution2D(self.embed_dim,128,1), # 16->128という全結合が10(=ss)個あり、重みが全部同じ
            last2 = L.Convolution2D(128,self.a_dim,1), 
            embedId = L.EmbedID(self.mu+1,self.embed_dim),

            lastlinear = L.Linear(self.a_dim,self.a_dim),
        )
        dlt = [2**i for i in range(9+1)]
        for i,d in enumerate(dlt):
            self.add_link('dc1-{}'.format(i), L.DilatedConvolution2D(None, self.a_dim, (1, 3), pad=(0,d), dilate=d))
            self.add_link('c1-{}'.format(i), L.Convolution2D(self.a_dim, self.a_dim, 1))
            self.add_link('dc2-{}'.format(i), L.DilatedConvolution2D(self.a_dim, self.a_dim, (1, 3), pad=(0,d), dilate=d))
            self.add_link('c2-{}'.format(i), L.Convolution2D(self.a_dim, self.a_dim, 1))
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)
    def preprocess(self, lot): # 入力データの前処理をこいつで規定する
        return quantize_linear(lot)        
    def __call__(self, input_data, mode='train'):
        '''
        input_data <Variable (bs, ssize*slen) int32> 0〜255
        returns <Variable (bs, 128(=self.a_dim), ssize) float32> 0〜1で出して。
        '''       
        #assert(input_data.shape[1] % 512==0)
        #ssize = input_data.shape[1]//512
        #assert(ssize == self.ssize)
        bs = input_data.shape[0]
        xp = cuda.get_array_module(input_data.data)

        h = F.reshape(F.transpose(input_data, (1,0)), (self.ssize*self.slen, bs, 1))
        if (self.embed_dim>1): # EmbedIDする
            # print("input_data.shape:",input_data.shape) # (5120, bs, 1)
            h = self.embedId(h)
            # print("h.shape:",h.shape) # (5120, bs, 1, 16)
            h = F.transpose(h, (1, 3, 2, 0))
            # print("h.shape:",h.shape) # (bs, 16, 1, 5120)
        else: # EmbedIDしない。1次元そのまま
            h = chainer.Variable(h.data.astype(xp.float32)) # int32をfloat32にする
            h = F.transpose(h, (1, 2, 0)) # (bs, 1, 5120)
            h = F.reshape(h, (h.shape[0], 1, h.shape[1], h.shape[2]))
            # 前処理
            h = h / 128 - 1.
        # h.shape == (bs, embed_dim, 1, ssize*512)

        #それぞれの層の出力をこのoutに足し合わせる（これはwavenetと同じような仕様）
        out = chainer.Variable(xp.zeros((h.shape[0], self.a_dim, h.shape[2], h.shape[3]),dtype=xp.float32))
        #dilationを2回積み重ねる
        for i in range(10):
            _h = self['dc1-{}'.format(i)](h)
            _h = F.tanh(_h)*F.sigmoid(_h)
            _h = self['c1-{}'.format(i)](_h)
            out += _h
            if (i>0):
                h = h+_h
            else:
                h = _h

        for i in range(10):
            _h = self['dc2-{}'.format(i)](h)
            _h = F.tanh(_h)*F.sigmoid(_h)
            _h = self['c2-{}'.format(i)](_h)
            out += _h
            h = h+_h

        # print('*',out.shape) # (bs, a_dim, 1, self.ssize*self.slen)

        if (self.embed_dim==1):
            out = F.average_pooling_2d(out,(1,self.ssize*self.slen)) # (bs, a_dim, 1, 1)
            out = self.lastlinear(out)
            out = F.sigmoid(out)
            out = F.reshape(out, (bs, self.a_dim, self.ssize))
        else:
            #512サンプルのフレームで取りまとめて、0~1で出力する
            out = F.average_pooling_2d(out,(1,512))
            out = F.relu(out)
            # print('*',out.shape) # (bs, embed_dim, 1, ssize)
            out = self.last1(out)
            out = F.relu(out)
            # print('**',out.shape) # (bs, 128, 1, ssize)
            out = self.last2(out)
            out = F.sigmoid(out)
            # print('***',out.shape) # (bs, a_dim, 1, ssize)
            out = F.reshape(out, (bs, self.a_dim, self.ssize))

        return out


class TransNet3:#(chainer.Chain):
    '''
    耳コピ用のクラスです．fmdlにフォワードモデル(chainer.Chain)を指定。
    音声データはmu=255のmu-lawエンコードした後のデータをonehotベクトルで入力します
    出力データは128個の音階の0～1データです．

    fmdl.__call__(x) は
    x <Variable (bs, ssize*512)> int32
    returns <Variable (bs, 128(=self.a_dim), ssize)> float32
    となってないといけない。
    （と書いてあるが、このクラス内のコードではこれらを一切要請していない。
    update()のxとt、error()のinput_dataとoutput_dataの型が
    fmdlの入出力フォーマットと一致してさえいれば良い。）
    '''
    # def __init__(self, embed_dim=16, a_dim=128):
    def __init__(self, fmdl, a_nn=list(range(128))):
        # self.mu = 255#μ-rawのμです
        # self.train_sample = 2**9#512サンプルの1出力です
        # self.total_iter = 0
        # self.loss = None
        # self.a_dim = a_dim # 出力の次元
        # self.embed_dim = embed_dim

        self.fmdl = fmdl
        # self.fmdl = wavenet(embed_dim=embed_dim, a_dim=a_dim) # フォワード計算を外に出そう
        # self.fmdl = yosnet(embed_dim=embed_dim, a_dim=a_dim) # フォワード計算を外に出そう
        self.a_nn = a_nn

        self.lossfrac = np.zeros(2) # loss平均計算用
        self.acctable = np.zeros((2, 2),dtype=np.int64) # 正解率平均計算用
        self.optimizer = optimizers.Adam()
        # self.optimizer.setup(self)
        self.optimizer.setup(self.fmdl)
    def cleargrads(self):
        self.fmdl.cleargrads()
    def error(self, input_data, output_data, mode='train'):
        '''
        input_data <Variable (bs, ssize*512)> int32
        output_data <Variable (bs, 128(=self.a_dim), ssize)> float32
        '''
        myoutput_data = self.fmdl(input_data, mode=mode)
        # print("myoutput_data.data---------------------------------")
        # print(myoutput_data.data.reshape((30,10))) # (30, 1, 1, 10) # ほとんど全部同じ値を出力している・・・


        # err = F.mean_squared_error(myoutput_data, output_data)
        # 素子ごとに2値分類してると考えて、cross entropy を考えよう
        err = -F.log(1e-14 + F.absolute(1. - output_data - myoutput_data))
        err = F.sum(err)/err.data.size

        self.lossfrac += np.array([tocpu(err.data), 1.])
        y = (tocpu(myoutput_data.data).flatten() > 0.5).astype(np.int64)
        t = (tocpu(output_data.data).flatten() > 0.5).astype(np.int64)
        _ = np.array([len(y), np.sum(y), np.sum(t), np.sum(y*t)])
        self.acctable += np.dot(np.array([[1,-1,-1,1],[0,0,1,-1],[0,1,0,-1],[0,0,0,1]]), _).reshape((2,2)) # いわゆる 2x2 表 (ネットワークの回答(0 or 1), 実際(0 or 1)) # TODO: このacctable更新のコードは超わかりづらい。高速かつわかり易く書き換えられないか？
        self.lastoutput = myoutput_data.data # array (bs, self.a_dim, ssize)
        self.lastanswer = output_data.data # array (bs, self.a_dim, ssize)
        self.logposterior = np.sum(np.sum(np.log(np.maximum(tocpu(F.absolute(1. - output_data - myoutput_data).data), 1e-14)), axis=2), axis=1) # array (bs)
        return err
    '''
    def set_training_data(self, train_in, train_out):
        self.train_in = train_in
        self.train_out = train_out
        self.n = train_out.shape[1]
    '''
    '''
    def learn(self,size): # （代わりにupdateを使ってます）（errorの引数フォーマット変えたので動かない）
        #適当なところから30*512サンプル分の教師データを抜き出す
        idx = np.random.randint(self.n-size)
        batch_in = self.train_in[idx*self.train_sample:(idx+size)*self.train_sample]
        batch_in = batch_in.reshape(self.train_sample*size,1,1)
        batch_in = chainer.Variable(batch_in)
        batch_out = self.train_out[:,idx:idx+size]
        batch_out = batch_out.reshape(1,self.a_dim,1,size)
        batch_out = batch_out.astype(np.float32)
        batch_out = chainer.Variable(batch_out)

        #更新
        self.cleargrads()
        self.loss = self.error(batch_in,batch_out)
        self.loss.backward()
        self.optimizer.update()
        self.total_iter += 1
    '''
    def update(self, x, t, mode="train"):
        '''
        入力xと出力t（1バッチ分）のVariableを放り込んで学習。
        mode=='test'ならloss計算のみ（学習なし）。
        x <Variable (bsize, ssize * 512) int32>
        t <Variable (bsize, 128(=self.a_dim), ssize) float32>
        '''
        self.loss = self.error(x, t, mode=mode)
        if (mode=='train'):
            self.cleargrads()
            self.loss.backward() # lossはscalarなのでいきなりこれでok
            self.optimizer.update()
        elif mode=='test':
            pass
        else:
            print("mode must be 'train' or 'test'.")
            raise ValueError
    def aveloss(self, clear=False):
        ret = self.lossfrac[0]/self.lossfrac[1]
        if (clear): self.lossfrac = np.zeros(2)
        return ret
    def getacctable(self, clear=False):
        ret = np.copy(self.acctable)
        if (clear): self.acctable = np.zeros((2,2),dtype=np.int64)
        return ret





