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

def make_cqt_inout(data_dir_or_data_list, file, mode='abs', scale_mode=None):
    '''
    wsdataからcqtで変換したwaveとscoreをnpzに格納します

    data_dir_or_data_list: wsdataが保存されたディレクトリ名(str) or wsdataファイル名のリスト(list of str)
    file: 出力ファイル
    mode: CQTからどんな値を抽出するか
        'abs' 絶対値(chl=1)
        'raw' 実部と虚部をそのままだす(chl=2)
    scale_mode: cqtの時のスケールの設定
        None   librosa.core.cqtのデフォルト設定 (C1 ~= 32.70 Hz (#24) から 84鍵)
        'midi' midiノートナンバーの #0 -- #127

    npz内データ
    spect np.narray [chl, pitch, seqlen]
    score np.narray [pitch, seqlen]
    '''
    assert mode=='abs' or mode=='raw'

    if scale_mode is None:
        fmin, n_bins = None, 84
    elif scale_mode == 'midi':
        fmin, n_bins = 440 * 2**(-69/12), 128

    if isinstance(data_dir_or_data_list, str):
        path_list = glob.glob("%s/*.wsd"%data_dir_or_data_list)
    else:
        path_list = data_dir_or_data_list
    spect,score = [],[]
    for path in path_list:
        print('processing: wsdata =',path)
        data = load_wsdata(path)

        n_bins_limit = 120
        sr = 44100 # musicNetのサンプリングレートは44100Hzです
        if n_bins <= n_bins_limit:
            cqt_array = np.expand_dims(librosa.core.cqt(data.wave,sr=sr,fmin=fmin,n_bins=n_bins), axis=0)
        else: # hop_length==512 だと一度に10オクターブ分までしかできないらしいので．
            lower = librosa.core.cqt(data.wave,sr=sr,fmin=fmin,n_bins=n_bins_limit)
            upper = librosa.core.cqt(data.wave,sr=sr,fmin=fmin*2**(n_bins_limit/12),n_bins=n_bins-n_bins_limit)
            cqt_array = np.expand_dims(np.r_[lower, upper], axis=0)

        if mode == 'abs':
            spect_ = np.abs(cqt_array).astype(np.float32)
        elif mode == 'raw':
            spect_ = np.concatenate([cqt_array.real, cqt_array.imag], axis=0).astype(np.float32)
        else:
            raise ValueError

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

def make_cqt_input(file , mode='abs', scale_mode=None):
    '''
    waveファイルからcqtで変換します．

    file: wavファイル名
    mode: CQTからどんな値を抽出するか
        'abs' 絶対値(chl=1)
        'raw' 実部と虚部をそのままだす(chl=2)
    scale_mode: cqtの時のスケールの設定
        None   librosa.core.cqtのデフォルト設定 (C1 ~= 32.70 Hz (#24) から 84鍵)
        'midi' midiノートナンバーの #0 -- #127

    ret: np.narray [chl, pitch, seqlen]
    '''
    assert mode=='abs' or mode=='raw'

    sr, data = sp.io.wavfile.read(file)
    data = data.astype(np.float64)
    data = data.mean(axis=1)
    data /= np.abs(data).max()

    if scale_mode is None:
        fmin, n_bins = None, 84
    elif scale_mode == 'midi':
        fmin, n_bins = 440 * 2**(-69/12), 128

    n_bins_limit = 120
    if n_bins <= n_bins_limit:
        cqt_array = np.expand_dims(librosa.core.cqt(data,sr=sr,fmin=fmin,n_bins=n_bins), axis=0)
    else: # hop_length==512 だと一度に10オクターブ分までしかできないらしいので．
        lower = librosa.core.cqt(data,sr=sr,fmin=fmin,n_bins=n_bins_limit)
        upper = librosa.core.cqt(data,sr=sr,fmin=fmin*2**(n_bins_limit/12),n_bins=n_bins-n_bins_limit)
        cqt_array = np.expand_dims(np.r_[lower, upper], axis=0)

    if mode == 'abs':
        return np.abs(cqt_array).astype(np.float32)
    elif mode == 'raw':
        return np.concatenate([cqt_array.real, cqt_array.imag], axis=0).astype(np.float32)
    else:
        raise ValueError
