#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:19:26 2016

@author: marshi
"""

import numpy as np
import pretty_midi

def midi_to_output(midi_file,out_file,wav_fre = 44100,train_sample = 512.0 ):
    '''
    midiファイルを出力形式にします
    wav_fre:Waveファイルの周波数(Hz)
    train_sample:何サンプルで教師データを用意するか
    '''
    
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    train_out = midi_data.get_piano_roll(wav_fre/train_sample)
    
    #音の大きさに関わらず音が出てれば、1とする
    train_out = np.sign(train_out)
    np.save(out_file,train_out)