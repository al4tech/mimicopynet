#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:25:35 2016

@author: marshi
"""

import numpy as np
import mimicopynet as mcn
import glob
from scipy.fftpack import fft,ifft
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import matplotlib.pyplot as plt
import librosa

#mcn.data.piano_train_data("musicnet.npz", "musicnet_metadata.csv", "train_data")
#mcn.data.solo_piano_to_wsdata("musicnet.npz", "musicnet_metadata.csv", "wsdata")
model = mcn.model.CNN()
model.load_cqt_inout("testdata.npz")
model.learn()
