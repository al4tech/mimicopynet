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

def musicnet_to_wsdata(file, meta, out_dir, ensemble=None):
    '''
    musicnetのピアノ曲をwsdataに変換します

    arg
    file: musicnet.npzのパス
    meta: musicnet_metadata.csvのパス
    out_dir: wsdataファイルを保存するディレクトリ
    ensemble: どの楽器編成の曲を，wsdataにするか ex:"Solo Piano"
            Noneの時は全部
             (楽器編成を指定する文字列については musicnet_metadata.csv 参照)
    '''
    data = np.load(open(file,'rb'),encoding='latin1', allow_pickle=True)
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(meta)
    if ensemble is None:
        ids = meta["id"].astype(str).tolist()
    else:
        ids = meta[meta['ensemble']==ensemble]["id"].astype(str).tolist()
    for h,id in enumerate(ids):#idは数値ではなく文字列
        wsdata = wavescoredata()
        print('processing: id =',id,'(',h+1,'/',len(ids),')')
        x, y = data[id] # x: 波形 (ndarray<float: -1~1> (num_of_sample,))
                        # y: 楽譜データ (intervaltree)
        in_npy = x
        intvl = 512
        out_sample = np.array(range(0,len(x),intvl))
        out_npy = np.zeros((128, len(out_sample)))
        # outを集計
        for i,s in enumerate(out_sample):
            nns = [n[2][1] for n in y[s]]
            for nn in nns:
                out_npy[nn,i] = 1.
        print('    .wave.shape:', in_npy.shape, '.score.shape:', out_npy.shape, 'score_sample.shape:', out_sample.shape)
        print('    .wave.dtype:', in_npy.dtype, '.score.dtype:', out_npy.dtype, 'score_sample.dtype:', out_sample.dtype)
        wsdata.wave = in_npy
        wsdata.score = out_npy
        wsdata.score_sample = out_sample
        # TODO: 複数パートの情報を込めるべきか？ ← 込めた方が良い．.wave と .score に一次元加える？
        savefilename = out_dir+'/'+str(id)+'.wsd'
        wsdata.save(savefilename)
        print('    saved:',savefilename)
    print('Done.')
