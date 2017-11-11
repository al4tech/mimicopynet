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

    score: np.narray [pitch, seqlen] 各要素は 0 or 1 であること。 
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

def score_to_image(score, imgfile):
    '''
    音がなっているかのscoreデータから
    ピアノロール画像を生成します

    score: np.narray [pitch, seqlen] 各要素は 0 以上 1 以下にすること推奨
    imgfile: 出力ファイル名　(pdfからまず対応予定)
    '''
    import matplotlib.pyplot as plt
    width = 2048

    padded_score = np.zeros((score.shape[0], (score.shape[1]+width-1)//width*width)) + np.nan
    padded_score[:,:score.shape[1]] = score
    score = padded_score

    num_row = score.shape[1]//width
    fig = plt.figure(figsize=(20,4+2*num_row))
    for i in range(num_row):
        plt.subplot(num_row, 1, 1+i)
        plt.imshow(score[:,i*width:(i+1)*width], origin='lower')
    plt.subplots_adjust(bottom=0.02, left=0.05, right=0.95, top=0.98, hspace=0.0001)
    plt.savefig(imgfile)













