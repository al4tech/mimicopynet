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

class TransNet2(chainer.Chain):
    '''
    耳コピ用のWavenetです．
    音声データはmu=255のmu-lawエンコードした後のデータをonehotベクトルで入力します
    出力データは128個の音階の0～1データです．
    '''
    def __init__(self):
        super(TransNet2, self).__init__(
            last1 = L.Convolution2D(16,128,1),
            last2 = L.Convolution2D(128,128,1),
            embedId = L.EmbedID(256,16),
        )
        dlt = [1,2,4,8,16,32,64,128,256,512]
        for i,d in enumerate(dlt):
            self.add_link('dc1-{}'.format(i), L.DilatedConvolution2D(16, 32, (1, 3), pad=(0,d), dilate=d))
            self.add_link('c1-{}'.format(i), L.Convolution2D(32, 16, 1))
            self.add_link('dc2-{}'.format(i), L.DilatedConvolution2D(16, 32, (1, 3), pad=(0,d), dilate=d))
            self.add_link('c2-{}'.format(i), L.Convolution2D(32, 16, 1))
    def __call__(self, input_data):
        h = self.embedId(input_data)
        h = F.transpose(h, (1, 3, 2, 0))

        #それぞれの層の出力をこのoutに足し合わせる（これはwavenetと同じような仕様）
        #out = chainer.Variable(np.zeros(h.shape,dtype=np.float32))

        #dilationを2回積み重ねる
        for i in range(10):
            _h = self['dc1-{}'.format(i)](h)
            _h = F.tanh(_h)*F.sigmoid(_h)
            _h = self['c1-{}'.format(i)](_h)
            #out += _h
            h = h+_h

        for i in range(10):
            _h = self['dc2-{}'.format(i)](h)
            _h = F.tanh(_h)*F.sigmoid(_h)
            _h = self['c2-{}'.format(i)](_h)
            #out += _h
            h = h+_h

        #512サンプルのフレームで取りまとめて、0~1で出力する
        out = F.average_pooling_2d(h,(1,512))
        out = F.relu(out)
        out = self.last1(out)
        out = F.relu(out)
        out = self.last2(out)
        out = F.sigmoid(out)
        return out
    def error(self, input_data, output_data):
        '''
        今回は二乗誤差としておきます
        '''
        err = F.mean_squared_error(self(input_data),output_data)
        return err
