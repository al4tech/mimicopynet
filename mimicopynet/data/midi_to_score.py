#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:19:26 2016

@author: marshi
"""

import numpy as np
import pretty_midi

"""
def midi_to_score_sum(midi_file, wav_fre=44100, train_sample=512.0, sampling_rate=None):
    '''
    [DEPRECATED]
    midiファイルをスコア形式(array)に変換して返します．

    Args:
        midi_file (str):
            （単一の）midiファイルへのパス．
        wav_fre (int or float): [DEPRECATED]
            Waveファイルの周波数(Hz)
        train_sample (int or float): [DEPRECATED]
            スコア形式の時間軸方向の1が，Waveファイルの何サンプル分に相当するか．
        sampling_rate (float or None):
            スコア形式の時間軸方向のスケール（1秒あたり，インデクスがいくつ増えるか．）．
            大きな値にするほど，返ってくる array のサイズは大きくなります．
            （これが None の場合は，sampling_rate := wav_fre / train_sample によって計算されます．
              None でない場合は，こちらの値が優先して用いられます．）

    Returns:
        (numpy.ndarray):
            スコア情報．各値はベロシティ値（0 以上 127 以下）がそのまま入る．
            音符が重なっている場合は，和が計算されるため，127 を超えることがある．
            shape = (128, length), dtype = np.float64
                ただし length == int(sampling_rate * 最後のmidiイベントの時刻[sec]) である．
    '''
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    if sampling_rate is None:
        sampling_rate = wav_fre / train_sample
    return midi_data.get_piano_roll(sampling_rate)
    # NOTE: get_piano_roll の docstring には 第一引数は int と書かれているが，実装を見ると float で問題なさそう．
"""

def midi_to_score(midi_file, sampling_rate, merge_part=None, mode='hold'):
    '''
    midiファイルをスコア形式(array)に変換して返します．
    ------
    Args:
        midi_file (str):
            （単一の）midiファイルへのパス．
        sampling_rate (float or int):
            スコア形式の時間軸方向のスケール（1秒あたり，インデクスがいくつ増えるか．）．
            大きな値にするほど，返ってくる array のサイズは大きくなります．
        merge_part (None or str):
            パートをマージするかどうかのオプション．下参照．
        mode (str):
            'hold' or 'onset'

    Returns:
        (numpy.ndarray):
            スコア情報．

            mode == 'hold' の場合:
                これの時間方向の i 番目の要素が正になる必要十分条件は，厳密には，
                「(i+1)/sr 未満の時刻に鳴り始め，(i+1)/sr 以上の時刻に鳴り終わる」
                正のベロシティのノーツが存在することです．
                返り値には，ベロシティ値（0 以上 127 以下）がそのまま入ります．
                shape=(パート数, 128, length), dtype=np.float32

            mode == 'onset' の場合:
                これの時間方向の i 番目の要素が正になる必要十分条件は，厳密には，
                「i/sr 以上，(i+1)/sr 未満の時刻に鳴り始める」
                正の発音時間を持つノーツが存在することです．
                返り値には，発音時間 [sec] が入ります．
                shape=(パート数, 128, length), dtype=np.float32

                ただし length == int(sampling_rate * 最後のMIDIイベントの時刻[sec]) です．

            merge_part はデフォルトでは None ですが，'sum' or 'max' の場合
                返り値の 0 次元めが sum または max で集約されます
                [DEPRECATED]
                ．
    '''
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    # If there are no instruments, return an empty array
    if len(midi_data.instruments) == 0:
        return np.zeros([0, 128, 0], dtype=np.float32)

    # Get piano rolls for each instrument
    if mode == 'hold':
        piano_rolls = [i.get_piano_roll(fs=sampling_rate).astype(np.float32)
                       for i in midi_data.instruments]
    elif mode == 'onset':
        piano_rolls = []
        for inst in midi_data.instruments:
            pr = np.zeros([128, int(sampling_rate * inst.get_end_time())+1], dtype=np.float32)
            for note in inst.notes:
                pr[note.pitch, int(sampling_rate * note.start)] = note.end - note.start
            piano_rolls.append(pr)

    # Allocate piano roll,
    # number of columns is max of # of columns in all piano rolls
    piano_roll = np.zeros([len(piano_rolls), 128, np.max([p.shape[1] for p in piano_rolls])], dtype=np.float32)

    # aggregate piano roll
    for i, roll in enumerate(piano_rolls):
        piano_roll[i, :, :roll.shape[1]] = roll

    if merge_part:
        if merge_part == 'sum':
            piano_roll = np.sum(piano_roll, axis=0)
        elif merge_part == 'max':
            piano_roll = np.max(piano_roll, axis=0)
        else:
            raise ValueError

    return piano_roll    


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