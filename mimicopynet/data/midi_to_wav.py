#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:21:15 2016

@author: marshi
"""

from midi2audio import FluidSynth
import os

def midi_to_wav(midi_file,wav_file):
    sf = os.path.join(os.path.dirname(__file__), "TimGM6mb.sf2")
    fs = FluidSynth(sf)
    fs.midi_to_audio(midi_file, wav_file)