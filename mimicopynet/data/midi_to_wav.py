#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:21:15 2016

@author: marshi
"""

# from midi2audio import FluidSynth
import os


import subprocess
DEFAULT_SOUND_FONT = '~/.fluidsynth/default_sound_font.sf2'
DEFAULT_SAMPLE_RATE = 44100

class FluidSynth():
    def __init__(self, sound_font=DEFAULT_SOUND_FONT, sample_rate=DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.sound_font = os.path.expanduser(sound_font)

        # fluidsynth のパスを環境変数から取得する．（設定されていない場合は 'fluidsynth' とする．）
        if 'FLUIDSYNTH_PATH' in os.environ:
            self.fluidsynth_path = os.environ['FLUIDSYNTH_PATH']
        else:
            self.fluidsynth_path = 'fluidsynth'

    def midi_to_audio(self, midi_file, audio_file):
        dev_null = open(os.devnull, 'w')
        subprocess.call([self.fluidsynth_path, '-ni', self.sound_font, midi_file, '-F', audio_file, '-r', str(self.sample_rate)],
            stdout=dev_null)

    def play_midi(self, midi_file):
        subprocess.call([self.fluidsynth_path, '-i', self.sound_font, midi_file, '-r', str(self.sample_rate)],
        stdout=dev_null)



def midi_to_wav(midi_file,wav_file):
    sf = os.path.join(os.path.dirname(__file__), "TimGM6mb.sf2")
    fs = FluidSynth(sf)
    fs.midi_to_audio(midi_file, wav_file)