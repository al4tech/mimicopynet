#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:18:17 2016

@author: marshi
"""
import numpy as np
import pandas as pd
import os

def piano_train_data(file, meta, out_dir):
    train_data = np.load(open(file,'rb'),encoding='latin1')
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(meta)
    ids = meta[meta['ensemble']=="Solo Piano"]["id"].astype(str).tolist()
    for id in ids:#idは文字列
        print('processing: id =',id)
        x, y = train_data[id] # x: 波形 (ndarray<float: -1~1> (num_of_sample,))、y: 楽譜データ (intervaltree)
        in_npy = x
        intvl = 512
        out_sample = np.array(range(0,len(x),intvl))
        out_npy = np.zeros((128, len(out_sample)))
        # outを集計
        for i,s in enumerate(out_sample):
            nns = [n[2][1] for n in y[s]]
            for nn in nns:
                out_npy[nn,i] = 1.
        print(in_npy.shape, out_npy.shape, out_sample.shape)
        np.savez(out_dir+'/'+str(id)+'.npz', wave=in_npy, score=out_npy, score_sample=out_sample)
