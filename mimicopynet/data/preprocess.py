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

def make_cqt_inout(data_dir, file, mode='abs'):
    '''
    wsdataからcqtで変換したwaveとscoreをnpzに格納します

    data_dir: wsdataが保存されたディレクトリ
    file: 出力ファイル
    model: CQTからどんな値を抽出するか
        'abs' 絶対値
        'raw' 実部と虚部をそのままだす

    npz内データ
    spect np.narray [chl, pitch, seqlen]
    score np.narray [pitch, seqlen]
    '''
    assert mode=='abs' or mode=='raw'

    spect,score = [],[]
    for path in glob.glob("%s/*.wsd"%data_dir):
        print('processing: wsdata =',path)
        data = load_wsdata(path)
        if mode == 'abs':
            spect_ = np.abs(librosa.core.cqt(data.wave))
            spect_ = np.expand_dims(spect_, axis=0)
        elif mode == 'raw':
            spect_ = librosa.core.cqt(data.wave)
            spect_ = np.expand_dims(spect_, axis=0)
            spect_ = np.concatenate([spect_.real, spect_.imag], axis=0)
        score_ = data.score
        length = min([spect_.shape[2],score_.shape[1]])
        spect_, score_ = spect_[:,:,:length], score_[:,:length]
        print(spect_.shape, score_.shape)
        spect.append(spect_)
        score.append(score_)
    spect = np.concatenate(spect, axis=2)
    score = np.concatenate(score, axis=1)
    print("shape",spect.shape, score.shape)
    np.savez(file, spect=spect, score=score)
