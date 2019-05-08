#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:18:17 2016

@author: yos1up
"""
import numpy as np
import pandas as pd
import os
import h5py

from . import WSH5

def musicnet_to_wsh5(file, meta, out_dir, ensemble=None):
    '''
    musicnet の曲を wsh5 形式 (wave-score-h5) に変換します

    musicnet の中には「4ケタのID番号」でナンバリングされた曲が数百曲収録されています．
    それらを {out_dir}/{ID}.h5 という名前の wsh5 ファイルに変換していきます． 

    ------
    Args:
        file (str):
            musicnet.npzのパス
        meta (str):
            musicnet_metadata.csvのパス
        out_dir (str):
            wsh5 ファイルたちを保存するディレクトリ
        ensemble (str or None):
            どの楽器編成の曲を，wsh5 にするか． 例: "Solo Piano"
            Noneの時は，全曲を wsh5 にします．
             (楽器編成を指定する文字列については musicnet_metadata.csv 参照)
    '''
    data = np.load(open(file,'rb'), encoding='latin1', allow_pickle=True)
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(meta)
    if ensemble is None:
        ids = meta["id"].astype(str).tolist()
    else:
        ids = meta[meta['ensemble']==ensemble]["id"].astype(str).tolist()
    for h, id_str in enumerate(ids):
        # wsdata = wavescoredata()
        print('processing: id =',id_str,'(',h+1,'/',len(ids),')')
        x, y = data[id_str] # x: 波形 (ndarray<float: -1~1> (num_of_sample,))
                            # y: 楽譜データ (intervaltree)
        in_npy = x.astype(np.float32).reshape(1, -1) # [パート，時間]
        intvl = 512
        out_sample = np.arange(0, len(x), intvl, dtype=np.int64)
        out_npy = np.zeros([1, 128, len(out_sample)], dtype=np.float32) # [パート，音高，時刻]
        # NOTE: 音量情報も込められうるように int ではなく float にしておく．

        # outを集計
        for i,s in enumerate(out_sample):
            nns = [n[2][1] for n in y[s]]
            for nn in nns:
                out_npy[0, nn, i] = 1.
        print('    .wave.shape:', in_npy.shape, '.score.shape:', out_npy.shape, 'score_sample.shape:', out_sample.shape)
        print('    .wave.dtype:', in_npy.dtype, '.score.dtype:', out_npy.dtype, 'score_sample.dtype:', out_sample.dtype)


        savefilename = out_dir + '/' + str(id_str) + '.h5'
        WSH5.save_wsh5(savefilename, wave=in_npy, score=out_npy, score_sample=out_sample)
        print('    saved:', savefilename)
    print('Done.')
