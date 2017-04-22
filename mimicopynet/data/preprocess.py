#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:18:17 2016

@author: marshi
"""
import numpy as np
import os
import glob
import librosa
from .wavescoredata import load_wsdata

def make_cqt_inout(data_dir, file):
    '''
    wsdataからcqtで変換したwaveとscoreをnpzに格納します

    data_dir: wsdataが保存されたディレクトリ
    file: 出力ファイル
    '''
    spect,score = [],[]
    for path in glob.glob("%s/*.wsd"%data_dir):
        data = load_wsdata(path)
        spect_ = np.abs(librosa.core.cqt(data.wave))
        score_ = data.score
        length = min([spect_.shape[1],score_.shape[1]])
        spect_, score_ = spect_[:,:length], score_[:,:length]
        spect.append(spect_)
        score.append(score_)
    spect = np.concatenate(spect, axis=1)
    score = np.concatenate(score, axis=1)
    print(spect.shape, score.shape)
    np.savez(file, spect=spect, score=score)
