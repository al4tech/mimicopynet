
import numpy as np
import os
import h5py
import tempfile
from scipy.io import wavfile

import pretty_midi

from . import FluidSynth, midi_to_score, WSH5



def read_wave_in_float32(filename):
    """
    wave を読み込み，[-1, 1] に収まる float32 形式で返す．
    """
    sr, wave = wavfile.read(filename)
    if wave.dtype == np.int16:
        wave = wave.astype(np.float32) / 32768
    elif wave.dtype == np.int32:
        wave = wave.astype(np.float32) / 2147483648
    elif wave.dtype == np.float32:
        pass
    elif wave.dtype == np.uint8:
        wave = (wave.astype(np.float32) - 128) / 128
    return sr, wave


def midi_to_wsh5(midi_file, wsh5_file, sound_font):
    """
    Convert a midi file to a wsh5-formatted file.

    ------
    Args:
        midi_file (str):
            path to a midi file

        wsh5_file (str):
            name for saving. recommended extension: '.h5'

        sound_font (str):
            path to .sf2 file

    Examples:
        midi_to_wsh5('sounds/yos.mid', '_hoge.h5', sound_font='mimicopynet/soundfonts/TimGM6mb.sf2')
    """

    # midi ファイルから，1パートごとのみonになった midi ファイルを生成する．
    os.makedirs('_', exist_ok=True)
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    num_inst = len(midi_data.instruments)
    for i in range(num_inst):
        out = pretty_midi.PrettyMIDI()
        out.instruments.append(midi_data.instruments[i])
        out.write('_/part{}.mid'.format(i))

    # それらを wave に変換！
    fls = FluidSynth(sound_font=sound_font)
    waves = []
    for i in range(num_inst):
        # with tempfile.NamedTemporaryFile() as tf:
        fls.midi_to_audio('_/part{}.mid'.format(i), '_/part{}.wav'.format(i)) # 一時ファイルに wave を書き出す
        sr, wave = read_wave_in_float32('_/part{}.wav'.format(i))
        wave = wave.mean(axis=1) # モノラルに
        waves.append(wave) # wave: 1-dim

    # wave たちの情報を，単一の array，merged_wave に格納する．
    merged_wave = np.zeros([num_inst, np.max([len(w) for w in waves])], dtype=np.float32)
    for i, w in enumerate(waves):
        merged_wave[i,:len(w)] = w

    # score 情報の取得
    score = midi_to_score(midi_file, sampling_rate=44100/512)
    score_onset = midi_to_score(midi_file, sampling_rate=44100/512, mode='onset')
    score_sample = np.arange(score.shape[-1]) * 512

    # 音色情報
    insts = []
    for i, inst in enumerate(midi_data.instruments):
        insts.append({'program':inst.program, 'is_drum':inst.is_drum, 'name':inst.name})

    # 以上の情報を，単一の h5 ファイルに保存する．
    WSH5.save_wsh5(
        wsh5_file,
        wave=merged_wave,
        wave_sr=44100,
        score=score,
        score_onset=score_onset,
        score_sample=score_sample,
        inst_program=np.array([i['program'] for i in insts]),
        inst_is_drum=np.array([i['is_drum'] for i in insts]),
        # inst_name=np.array([i['name'].encode('ascii') for i in insts]),
    )

    """
    # 確認用
    summed_wave = np.mean(merged_wave, axis=0)
    summed_wave /= np.max(np.abs(summed_wave))
    wavfile.write('_/summed.wav', 44100, summed_wave)

    # 確認用2
    fls.midi_to_audio(midi_file, '_/original_waved.wav')
    """










