#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:18:17 2016

@author: marshi
"""
import numpy as np
import pandas as pd
import os
from .wavescoredata import wavescoredata

def solo_piano_to_wsdata(file, meta, out_dir):
    '''
    musicnetのピアノ曲をwsdataに変換します

    arg
    file: musicnet.npzのパス
    meta: musicnet_metadata.csvのパス
    out_dir: wsdataファイルを保存するディレクトリ
    '''
    data = np.load(open(file,'rb'),encoding='latin1')
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(meta)
    ids = meta[meta['ensemble']=="Solo Piano"]["id"].astype(str).tolist()
    for id in ids:#idは文字列
        wsdata = wavescoredata()
        print('processing: id =',id)
        x, y = data[id] # x: 波形 (ndarray<float: -1~1> (num_of_sample,))、y: 楽譜データ (intervaltree)
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
        wsdata.wave = in_npy
        wsdata.score = out_npy
        wsdata.score_sample = out_sample
        wsdata.save(out_dir+'/'+str(id)+'.wsd')
