#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:25:35 2016

@author: marshi
"""

import numpy as np
import mimicopynet

mimicopynet.data.make_random_song("train.mid")
mimicopynet.data.midi_to_wav("train.mid","train.wav")
mimicopynet.data.midi_to_output("train.mid","train_out.npy")
mimicopynet.data.wav_to_input("train.wav","train_in.npy")

train_in = np.load("train_in.npy")
train_out = np.load("train_out.npy")

model = mimicopynet.model.TransNet()

model.set_training_data(train_in,train_out)
for i in range(1000):
    model.learn(10)
    if i % 10 == 0:
        print(model.loss.data)