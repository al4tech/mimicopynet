#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:19:26 2016

@author: marshi
"""

import numpy as np
import pretty_midi

def midi_to_score(midi_file, wav_fre=44100, train_sample = 512.0):
    '''
    midiファイルをスコア形式(array)に変換して返します
    wav_fre:Waveファイルの周波数(Hz)
    train_sample:スコア形式の時間軸方向の1が，Waveファイルの何サンプル分に相当するか．
    '''
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    return midi_data.get_piano_roll(wav_fre/train_sample)
    # get_piano_roll の docstring には 第一引数は int と書かれているが，実装を見ると float で問題なさそう．
    # get_piano_roll の返り値の.shape[1] は int(第一引数 * 最後のmidiイベントの時刻) となるようだ．

def midi_to_output(midi_file,out_file,wav_fre = 44100,train_sample = 512.0 ):
    '''
    midiファイルをスコア形式(array)に変換し，それをdigitizeした上で out_file (.npy) に保存します
    wav_fre:Waveファイルの周波数(Hz)
    train_sample:何サンプルで教師データを用意するか（スコア形式の時間軸方向の1が，Waveファイルの何サンプル分に相当するか．）
    '''
    
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    train_out = midi_data.get_piano_roll(wav_fre/train_sample)
    
    #音の大きさに関わらず音が出てれば、1とする
    train_out = np.sign(train_out)
    np.save(out_file,train_out)