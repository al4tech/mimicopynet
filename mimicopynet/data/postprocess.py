#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:18:17 2016

@author: marshi
"""
import numpy as np
import os
import glob
import pretty_midi

def score_to_midi(score, midfile):
    '''
    音がなっているかのscoreデータから
    midiファイルを生成します

    score: np.narray [pitch, seqlen]
    midfile: midiファイル名
    '''
    pitch, time = np.where(score==1)
    dif = [time[i+1]-time[i] for i in range(len(pitch)-1)]
    end = np.concatenate([np.where(np.array(dif)!=1)[0],[len(time)-1]])
    start = np.concatenate([[0],end[:-1]+1])
    pitch = pitch[start]
    end = time[end]
    start = time[start]

    hz = 44100
    tempo = 120
    res = 960
    pm = pretty_midi.PrettyMIDI(resolution=res, initial_tempo=tempo)
    instrument = pretty_midi.Instrument(0)

    sample_t = (512/hz)
    for s_,e_,p_ in zip(start, end, pitch):
        note = pretty_midi.Note(velocity=100, pitch=p_, start=s_*sample_t,
                                end=e_*sample_t)
        instrument.notes.append(note)
    pm.instruments.append(instrument)
    pm.write(midfile)
