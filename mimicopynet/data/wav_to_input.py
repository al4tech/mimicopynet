#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:18:17 2016

@author: marshi
"""

import scipy as sp
from scipy.io import wavfile
import numpy as np

def wav_to_input(wav_file,in_file,bits=16,train_sample=512):
    '''
    waveファイルをμ-lawで変換して入力形式に変換します
    wavfile:wavファイル名
    train_in:入力形式のファイル名
    bits:wavファイルのbit数
    train_sample:何サンプルで出力データを用意するか
    '''

    wav = sp.io.wavfile.read(wav_file)
    wav = wav[1]
    
    #モノラル化
    wav = np.mean(wav,axis=1)
    
    #-1から1で正規化
    wav += 0.5
    wav /= 2**15-0.5
    
    #mu lawで変換
    mu = 255
    wav = np.sign(wav)*np.log(1+mu*np.abs(wav))/np.log(1+mu)
    wav = np.round((wav + 1) / 2 * mu).astype(np.int32)
    
    #train_sampleの定数倍個のサンプルにする
    last = int(wav.shape[0] - wav.shape[0]%train_sample)
    wav = wav[:last]
    
    np.save(in_file,wav)