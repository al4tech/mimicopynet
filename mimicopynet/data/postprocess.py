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

def digitize_score(score, algorithm='threshold'):
    '''
    pre-sigmoidで書かれた score から、0/1で書かれたscoreに変換します。

    score: np.ndarray [pitch, seqlen] 各要素はpre-sigmoidの実数値
    algorithm:変換アルゴリズムを指定する文字列

    returns: np.ndarray [pitch, seqlen] 各要素は 0/1
    '''
    if algorithm=='threshold':
        return (score > 0.) * 1
    elif algorithm=='mrf':
        seqlen = score.shape[1]
        minus_log_joint_prob = -np.log(np.array([[199.0, 1.0],[1.0, 99.0]])) # パラメータは要調整
        # [i,j]要素は、E(x_t==i and x_{t+1}==j)
        ret = np.zeros(score.shape).astype(np.int)

        for n in range(score.shape[0]):
            dp = np.zeros([2, seqlen])
            dp_memo = np.zeros([2, seqlen]) # for backtrack
            for t in range(seqlen):
                dp[0,t] = (dp[0,t-1] if t>0 else 0) + minus_log_joint_prob[0,0] + score[n,t]
                dp[1,t] = (dp[0,t-1] if t>0 else 0) + minus_log_joint_prob[0,1]
                tmp0 = (dp[1,t-1] if t>0 else 0) + minus_log_joint_prob[1,0] + score[n,t]
                tmp1 = (dp[1,t-1] if t>0 else 0) + minus_log_joint_prob[1,1]
                if dp[0,t] > tmp0:
                    dp[0,t] = tmp0; dp_memo[0,t] = 1
                if dp[1,t] > tmp1:
                    dp[1,t] = tmp1; dp_memo[1,t] = 1
            ret[n,seqlen-1] = np.argmin([dp[0,seqlen-1], dp[1,seqlen-1]])
            for t in range(score.shape[1]-1)[::-1]:
                ret[n,t] = dp_memo[ret[n,t+1],t+1]
        return ret
    else:
        raise NotImplementedError


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
    imgfile: 出力ファイル名 (plt.savefigが対応しているファイル名)
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













