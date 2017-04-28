#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:18:17 2016

@author: marshi
"""
import numpy as np
import scipy as sp
import glob
import librosa
from .wavescoredata import load_wsdata

def make_cqt_inout(data_dir, file, mode='abs'):
    '''
    wsdataからcqtで変換したwaveとscoreをnpzに格納します

    data_dir: wsdataが保存されたディレクトリ
    file: 出力ファイル
    mode: CQTからどんな値を抽出するか
        'abs' 絶対値(chl=1)
        'raw' 実部と虚部をそのままだす(chl=2)

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

def make_cqt_input(file , mode='abs'):
    '''
    waveファイルからcqtで変換します．

    file: wavファイル名
    mode: CQTからどんな値を抽出するか
        'abs' 絶対値(chl=1)
        'raw' 実部と虚部をそのままだす(chl=2)

    ret: np.narray [chl, pitch, seqlen]
    '''
    assert mode=='abs' or mode=='raw'

    data = sp.io.wavfile.read(file)[1]
    data = data.astype(np.float64)
    data = data.mean(axis=1)
    data /= np.abs(data).max()

    if mode == 'abs':
        input_data = np.abs(librosa.core.cqt(data)).astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)
    elif mode == 'raw':
        input_data = librosa.core.cqt(data)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.concatenate([input_data.real, input_data.imag],
                                    axis=0).astype(np.float32)
    return input_data
