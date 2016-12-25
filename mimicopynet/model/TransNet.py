# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:42:53 2016

@author: marshi
"""

import numpy as np
import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import scipy as sp

class TransNet(chainer.Chain):
    '''
    耳コピ用のWavenetです．
    音声データはmu=255のmu-lawエンコードした後のデータをonehotベクトルで入力します
    出力データは128個の音階の0～1データです．
    '''
    def __init__(self):
        self.mu = 255#μ-rawのμです
        self.train_sample = 2**9#512サンプルの1出力です
        self.total_iter = 0
        self.loss = None
        super(TransNet, self).__init__(
            last1 = L.Convolution2D(16,128,1),
            last2 = L.Convolution2D(128,128,1),
            embedId = L.EmbedID(self.mu+1,16),
        )
        dlt = [2**i for i in range(9+1)]
        for i,d in enumerate(dlt):
            self.add_link('dc1-{}'.format(i), L.DilatedConvolution2D(16, 32, (1, 3), pad=(0,d), dilate=d))
            self.add_link('c1-{}'.format(i), L.Convolution2D(32, 16, 1))
            self.add_link('dc2-{}'.format(i), L.DilatedConvolution2D(16, 32, (1, 3), pad=(0,d), dilate=d))
            self.add_link('c2-{}'.format(i), L.Convolution2D(32, 16, 1))
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self)
    def __call__(self, input_data):
        h = self.embedId(input_data)
        h = F.transpose(h, (1, 3, 2, 0))

        #それぞれの層の出力をこのoutに足し合わせる（これはwavenetと同じような仕様）
        out = chainer.Variable(np.zeros(h.shape,dtype=np.float32))

        #dilationを2回積み重ねる
        for i in range(10):
            _h = self['dc1-{}'.format(i)](h)
            _h = F.tanh(_h)*F.sigmoid(_h)
            _h = self['c1-{}'.format(i)](_h)
            out += _h
            h = h+_h

        for i in range(10):
            _h = self['dc2-{}'.format(i)](h)
            _h = F.tanh(_h)*F.sigmoid(_h)
            _h = self['c2-{}'.format(i)](_h)
            out += _h
            h = h+_h

        #512サンプルのフレームで取りまとめて、0~1で出力する
        out = F.average_pooling_2d(out,(1,512))
        out = F.relu(out)
        out = self.last1(out)
        out = F.relu(out)
        out = self.last2(out)
        out = F.sigmoid(out)
        return out
    def error(self, input_data, output_data):
        '''
        誤差は二乗誤差としておきます
        '''
        err = F.mean_squared_error(self(input_data),output_data)
        return err
    def set_training_data(self, train_in, train_out):
        self.train_in = train_in
        self.train_out = train_out
        self.n = train_out.shape[1]
    def learn(self,size):
        #適当なところから30*512サンプル分の教師データを抜き出す
        idx = np.random.randint(self.n-size)
        batch_in = self.train_in[idx*self.train_sample:(idx+size)*self.train_sample]
        batch_in = batch_in.reshape(self.train_sample*size,1,1)
        batch_in = chainer.Variable(batch_in)
        batch_out = self.train_out[:,idx:idx+size]
        batch_out = batch_out.reshape(1,128,1,size)
        batch_out = batch_out.astype(np.float32)
        batch_out = chainer.Variable(batch_out)

        #更新
        self.zerograds()
        self.loss = self.error(batch_in,batch_out)
        self.loss.backward()
        self.optimizer.update()
        self.total_iter += 1
