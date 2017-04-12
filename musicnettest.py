#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:25:35 2016

@author: marshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fs = 44100

meta = pd.read_csv("musicnet_metadata.csv")
ids = meta[meta['ensemble']=="Solo Piano"]["id"].astype(str).tolist()
#ids = ['1733','1734'...] 

musicnet_data = np.load(open('musicnet.npz','rb'),encoding='latin1')
X,Y = musicnet_data[ids[0]]
length = X.shape[0]/fs

stride = 512                         # 512 samples between windows
wps = fs/float(stride)               # ~86 windows/second
Yvec = np.zeros((int(length*wps),128))   # 128 distinct note labels

for window in range(Yvec.shape[0]):
    labels = Y[window*stride]
    for label in labels:
        Yvec[window,label.data[1]] = label.data[0]

fig = plt.figure(figsize=(20,5))
plt.imshow(Yvec.T,aspect='auto',cmap='ocean_r')
plt.show()

fig = plt.figure(figsize=(20,5))
plt.plot(np.arange(X.shape[0])/512,X)
plt.xlim(0,Yvec.shape[0])
plt.show()