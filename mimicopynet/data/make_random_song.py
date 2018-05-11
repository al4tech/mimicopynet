#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:20:17 2016

@author: marshi
"""

import numpy as np
import sys
import pretty_midi

class chord_generator():
    def __init__(self):
        self.dissonance = np.zeros([128, 128],dtype=np.float)
        for i in range(128):
            fi = 440*2**((i-69)/12)
            for j in range(i,128):
                fj = 440*2**((j-69)/12)
                for k in range(1,8):
                    for l in range(1,8):
                        self.dissonance[i,j] += 0.8**(k+l-2) * self.pure_tone_dissonance(fi*k, fj*l)
                # nオクターブ差の2音にはペナルティを与える（dissonance低い割に実際の音楽中での登場頻度は少ない気がするので）
                if i < j and (j - i)%12==0:
                    self.dissonance[i,j] *= 1
                # 最後にi,j入れ替えた側にコピー
                if i < j:
                    self.dissonance[j,i] = self.dissonance[i,j]
    def pure_tone_dissonance(self, f1, f2):
        if f1 > f2: f1, f2 = f2, f1
        return self.ndf_to_dissonance((f2-f1)/self.critical_bandwidth(f1))
    def critical_bandwidth(self, x):
        # TODO: 5000Hz以上は信憑性がない！
        # http://kogarashi.net/pitchblende/archives/9
        if x > 5000: return self.critical_bandwidth(5000) + 0.2 * (x - 5000)
        return (97.27500666976535 + x*(-0.10438654962335651 + x*(0.00013795361645053944 + x*(-7.50643237852489e-8 + (3.864047759875421e-11 - 2.715433778943346e-15*x)*x))))/(1 + x*(-0.0009358884960842764 + x*(4.5256275668615123e-7 + (3.595960994872951e-11 - 2.546811069035789e-15*x)*x)))
    def ndf_to_dissonance(self, x):
        # x >= 0 とする
        # http://kogarashi.net/pitchblende/archives/9
        if x > 1.5: return 0
        ret = (-0.00020410890928780975+x*(11.293912104998519 + x*(-64.80401525171715 + x*(142.52964847415188 + x*(-128.65558599526295 + 40.28982394173413*x)))))/(1 + x*(-1.1833111299032766 + x*(-4.205411534000957 + x*(0.8554237901431951 + 21.877444099945798*x))))
        return max(0, ret)
    def get_random_chord(self, pitch_range, num_notes_range, mode='uniform'):
        '''
        素朴な実装は下記のとおりであるが，これを，協和音を高い確率でサンプリングできるようにしたい．
        （協和音の方が耳コピの難易度は高いと考えられる）
        '''
        chord = np.random.choice(np.arange(pitch_range[0], pitch_range[1]+1),
                                np.random.randint(num_notes_range[0], num_notes_range[1]+1),
                                replace=False)
        if mode == 'uniform':
            pass
        elif mode == 'mcmc':
            if 2 <= len(chord) <= pitch_range[1] - pitch_range[0]:
                flg = np.zeros(128); flg[chord] = 1
                temp = 1.2
                for i in range(100):
                    while 1:
                        idx = np.random.randint(len(chord))
                        new_note = np.random.randint(pitch_range[0], pitch_range[1]+1)
                        if flg[new_note] == 0: break
                    diff = self.diff_chord_energy(chord, idx, new_note)
                    # print('    ', [pretty_midi.note_number_to_name(nn) for nn in sorted(chord)])
                    if diff < 0 or np.exp(-diff/temp) > np.random.rand(): # diff に応じて採択
                        flg[chord[idx]] -= 1; flg[new_note] += 1
                        chord[idx] = new_note
        else:
            raise ValueError
        return chord
    def chord_energy(self, chord):
        ret = 0
        for i in range(len(chord)):
            for j in range(i+1,len(chord)):
                ret += self.dissonance[chord[i], chord[j]]
        return ret
    def diff_chord_energy(self, chord, idx, new_note):
        ret = 0
        old_note = chord[idx]
        for i in range(len(chord)):
            if i == idx: continue
            ret += self.dissonance[new_note, chord[i]] - self.dissonance[old_note, chord[i]]
        return ret

'''
cg = chord_generator()
for i in range(100):
    print([pretty_midi.note_number_to_name(nn) for nn in sorted(cg.get_random_chord([60,72],[3,3],'mcmc'))])
import matplotlib.pyplot as plt
plt.imshow(cg.dissonance)
plt.show()
sys.exit()
'''


def make_random_song(file,lmb_start=0.6,lmb_stop=0.6,seed=None,tempo=120,res=960,
                     len_t=1000, inst=0):
    '''
    ポアソン過程でランダムな曲を作って、midiファイルに出力します。
    file <str> :出力midiファイル名
    lmb_start <float or list/array_of_float(len==128)> : note off から次の note on までの平均秒数
    lmb_stop <float or list/array_of_float(len==128)> : note on から note off までの平均秒数
    seed <int or None> : np.random.seedに渡す乱数の種．Noneなら初期化しない．
    tempo <int> : テンポ
    res <int> : レゾリューション
    len_t <float> : 生成される曲の秒数（の上限）
    inst <int or list_of_int> : 音色番号

    TODO: 複数の音色が混じるケース．同時に発音するケース(和音)，ドラム．
    '''
    if seed is not None: np.random.seed(seed)
    pm = pretty_midi.PrettyMIDI(resolution=res, initial_tempo=tempo) #pretty_midiオブジェクトを作ります

    if isinstance(inst, int): inst = [inst]

    cg = chord_generator()
    for ins in inst:
        # min_pitch_name, max_pitch_name = appropriate_pitch_range[ins]
        # min_pitch = pretty_midi.note_name_to_number(min_pitch_name) if isinstance(min_pitch_name, str) else min_pitch_name
        # max_pitch = pretty_midi.note_name_to_number(max_pitch_name) if isinstance(max_pitch_name, str) else max_pitch_name

        instrument = pretty_midi.Instrument(ins) #instrumentはトラックに相当します。
        # TODO: lmb_startとlmb_stopは数値であることをassertしたい
        t = 0.
        lmb_start = max(0.05, lmb_start)
        lmb_stop = max(0.05, lmb_stop)
        while t <= len_t:
            start = 0.05+np.random.exponential(lmb_start-0.05)+t
            stop = 0.05+np.random.exponential(lmb_stop-0.05)+start
            t = stop
            if t > len_t: break
            # 和音を決定する
            chord = cg.get_random_chord(appropriate_pitch_range[ins], appropriate_chord_num_notes[ins])
            for note_number in chord:
                note = pretty_midi.Note(velocity=np.random.randint(appropriate_velocity[ins][0], appropriate_velocity[ins][1]+1),
                                        pitch=note_number, start=start, end=stop) #noteはNoteOnEventとNoteOffEventに相当します。
                instrument.notes.append(note)
        pm.instruments.append(instrument)
    pm.write(file) #midiファイルを書き込みます。



# 楽器ごとに妥当な音域を定義する．TODO: 妥当な音価（音の長さ）も決める？
# http://www.page.sannet.ne.jp/hirasho/sound/gm-instruments.html
appropriate_pitch_range = [[21, 108], [21, 108], [21, 108], [21, 108], [28, 103], [28, 103], [41, 89], [36, 96], [60, 108], [72, 108], [53, 89], [53, 113], [48, 84], [65, 96], [60, 77], [60, 84], [36, 96], [36, 96], [36, 96], [21, 108], [36, 96], [53, 89], [60, 84], [53, 89], [40, 84], [40, 84], [40, 86], [40, 86], [40, 86], [40, 86], [40, 86], [40, 86], [28, 55], [28, 55], [28, 55], [28, 55], [28, 55], [28, 55], [28, 55], [28, 55], [55, 96], [48, 84], [36, 72], [28, 55], [28, 96], [28, 96], [23, 103], [36, 57], [28, 96], [28, 96], [36, 96], [36, 96], [48, 79], [48, 79], [48, 84], [48, 79], [58, 94], [34, 75], [29, 55], [58, 94], [41, 77], [36, 96], [36, 96], [36, 96], [54, 87], [49, 80], [42, 75], [37, 68], [58, 91], [52, 81], [34, 81], [50, 91], [74, 108], [60, 96], [60, 96], [60, 96], [60, 96], [55, 84], [60, 96], [60, 84], [21, 108], [21, 108], [48, 96], [48, 96], [48, 96], [48, 96], [36, 96], [21, 108], [36, 96], [36, 96], [36, 96], [48, 84], [48, 96], [21, 108], [36, 96], [36, 96], [36, 96], [36, 84], [60, 108], [36, 96], [36, 96], [36, 96], [36, 96], [36, 96], [48, 77], [48, 84], [50, 79], [55, 84], [48, 79], [36, 77], [55, 96], [48, 72], [72, 84], [60, 72], [52, 76], [48, 84], [48, 72], [36, 96], [36, 96], [48, 84], [48, 72], [48, 72], [48, 72], [48, 72], [48, 72], [48, 72], [48, 72], [48, 72]]
appropriate_velocity = [
    [64, 127], # 0 Piano
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 8 Mokkin
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [40, 80], # 16 Organ
    [40, 80],
    [40, 80],
    [40, 80],
    [40, 80],
    [40, 80],
    [40, 80],
    [40, 80],
    [64, 127], # 24 Guitar
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 32 Bass
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [40, 80], # 40 Violin
    [40, 80],
    [40, 80],
    [40, 80],
    [40, 80],
    [40, 80],
    [40, 80],
    [40, 80],
    [20, 40], # 48 String Ensemble
    [20, 40],
    [20, 40],
    [20, 40],
    [20, 40],
    [20, 40],
    [20, 40],
    [20, 40],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127], # 
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
    [64, 127],
]
appropriate_chord_num_notes = [
    [1, 4], # 0 Piano
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 2], # 8 Mokkin
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 3], # 16 Organ
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 4], # 24 Guitar
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 4],
    [1, 1], # 32 Bass
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 2], # 40 Violin
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 3], # 48 String Ensemble
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    # ここから先は未調査．
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1], # 
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
]


