#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:20:17 2016

@author: marshi
"""

import numpy as np
import pretty_midi

def make_random_song(file,lmb_start=50,lmb_stop=1,seed=0,tempo=120,res=960,len_t=1000):
    '''
    ポアソン過程でランダムな曲を作って、midiファイルに出力します。
    file <str> :出力midiファイル名
    lmb_start <float or list/array_of_float(len==128)> : note off から次の note on までの平均秒数
    lmb_stop <float or list/array_of_float(len==128)> : note on から note off までの平均秒数
    seed <int> : np.random.seedに渡す乱数の種
    tempo <int> : テンポ
    res <int> : レゾリューション
    len_t <float> : 生成される曲の秒数（の上限）
    '''
    np.random.seed(seed)
    pm = pretty_midi.PrettyMIDI(resolution=res, initial_tempo=tempo) #pretty_midiオブジェクトを作ります
    instrument = pretty_midi.Instrument(0) #instrumentはトラックに相当します。
    
    if (isinstance(lmb_start,int) or isinstance(lmb_start,float)):
        lmb_start = [lmb_start for i in range(128)]
    if (isinstance(lmb_stop,int) or isinstance(lmb_stop,float)):
        lmb_stop = [lmb_stop for i in range(128)]


    low_pitch = pretty_midi.note_name_to_number('A0')
    high_pitch = pretty_midi.note_name_to_number('A8')
    
    for note_number in range(low_pitch,high_pitch):
        t = 0.
        while t <= len_t:
            start = np.random.exponential(lmb_start[note_number])+t
            stop = np.random.exponential(lmb_stop[note_number])+start
            t = stop
            if t > len_t:
                break
            note = pretty_midi.Note(velocity=np.random.randint(50,127), pitch=note_number, start=start, end=stop) #noteはNoteOnEventとNoteOffEventに相当します。
            instrument.notes.append(note)
    
    
    pm.instruments.append(instrument)
    pm.write(file) #midiファイルを書き込みます。
