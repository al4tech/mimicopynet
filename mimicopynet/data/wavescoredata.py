#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:18:17 2016

@author: marshi
"""
import numpy as np
import pickle

class wavescoredata(object):
    '''
    ピアノ曲の音データと，楽譜データをnumpy形式でもつクラスです．

    wave: 音の波形データ
    score: wave512サンプルごとに対応する楽譜データ
    score_sample: それぞれのサンプルの最初のインデックス
    '''
    def save(self, file):
        '''
        pickle形式でセーブします
        '''
        with open(file, 'wb') as f:
            pickle.dump(self.__dict__, f)
    def load(self, file):
        '''
        ロードします
        '''
        with open(file, 'rb') as f:
            data = pickle.load(f)
        for k,v in data.items():
            self.__dict__[k] = v

def load_wsdata(file):
    wsdata = wavescoredata()
    wsdata.load(file)
    return wsdata
