#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:25:35 2016

@author: marshi, yos1up
"""

import numpy as np
import sys, time
from chainer import Chain, ChainList, cuda, gradient_check, Function, Link, optimizers, serializers, utils, Variable
from chainer import functions as F
from chainer import links as L
import mimicopynet

def export_graph(filename, vs):
    import chainer.computational_graph as ccg
    g = ccg.build_computational_graph(vs)
    with open(filename, 'w') as o:
        o.write(g.dump())

def in_npy_out_npy_to_lot(in_npy, out_npy, q_sample=5120, a_sample=10, a_nn=list(range(128)), posratio=None):
    '''
    list_of_tuple形式に変換します
    in_npy <np.array (S)> : 波形データ。ここでSはwaveのサンプル数
    out_npy <np.array (128, N)> : ピアノロールデータ。ここでNはピアノロールのサンプル数。
    a_nn <list of int> : ピアノロールのうち教師データとするノートナンバーのリスト（1音高だけ耳コピさせたい時は [60] などと指定）
    posratio <None or float> : 教師データが0ベクトルではないようなサンプルが全体のうちこの割合になるように調整する(どうやって？)

    各 i について、
    ピアノロールデータ out_npy　の　[i : i+a_sample] サンプル目　と、
    波形データ in_npy の round(S/N * (i + (a_sample-1)/2) - (q_sample-1)/2)サンプル目から連続するq_sampleサンプルと対応させます。
    TODO: これは妥当か？

    returns: list_of_tuple形式 各qはnp.array (r), 各aはnp.array (128)
    '''
    s = in_npy.shape[0]
    n = out_npy.shape[1]
    qa = []
    padded_in_npy = np.array([in_npy[0]] * q_sample + list(in_npy) + [in_npy[-1]] * q_sample)
    for i in range(n - a_sample):
        start = q_sample + int(np.round(s/n * (i+(a_sample-1)/2) - (q_sample-1)/2))
        qa.append((padded_in_npy[start:start+q_sample], out_npy[a_nn,i:i+a_sample]))

    if (posratio is not None):
        # 正例の割合を調整するために、負例を減らす
        # positive segment の個数を調整しよう
        pc = [np.count_nonzero(np.sum(d[1],axis=0)) for d in qa] # positive count
        sorted_idx = np.argsort(pc)[::-1]
        qa_new = []
        current_posratio = np.zeros(2)
        for i in sorted_idx: # positive segmentが多いデータから順に放り込んでいく。
            qa_new.append(qa[i])
            current_posratio += np.array([pc[i], a_sample])
            if (current_posratio[0]/current_posratio[1] < posratio): break
        print('target posratio:', posratio)
        print('achieved posratio:',current_posratio[0]/current_posratio[1])
        qa = qa_new
    # 最後にシャッフル
    np.random.shuffle(qa) # list_of_tuple形式は順不同とする (TODO: この点で、set_of_tupleの方が妥当？)
    return qa


'''
mimicopynet.data.make_random_song("train.mid")
mimicopynet.data.midi_to_wav("train.mid","train.wav")
mimicopynet.data.midi_to_output("train.mid","train_out.npy")
mimicopynet.data.wav_to_input("train.wav","train_in.npy")
'''
train_in = np.load("train_in.npy")
train_out = np.load("train_out.npy")
'''
mdl = mimicopynet.model.TransNet()
mdl.set_training_data(train_in, train_out)
while 1:
    for i in range(10):
        mdl.learn(size=10)
    print('aveloss', mdl.aveloss(clear=True))
    print('acctable', mdl.getacctable(clear=True))
'''
'''
rescale_input = False

# 0〜255を-1〜1にリスケール
if (rescale_input):
    train_in = train_in / 128 - 1.
#0 or 1
# train_out
'''


'''
モデルについて思うこと
・batch normalizationした方が良い?（batchsize100で1より収束CPU時間が遅くなっている？）
 ・音のうるささに関してだけEmbedIDは遅いだろうし意味はなさそう？（「入力の」1次元を16次元に展開する意味は？）
 ・softmax_cross_entropy的なlossのほうが良いだろう
 ・正例の個数を調節
 ・一音耳コピを
 ・バッチサイズあげて遅くなってるのはメモリのせいかも
　bs=100で10GBくらい食ってるけどそういうもんだっけ→取り急ぎ30に減らした
・ネットワークの要見直し
・高速化？ 20層は重たい・・・

・オンラインNMF？
'''


a_nn = [60] # list(np.arange(60, 72)) # 1音に限定した耳コピはどうだろう
a_dim = len(a_nn)
posratio = 0.5

np.random.seed(0)
lot = in_npy_out_npy_to_lot(train_in, train_out, q_sample=5120, a_sample=10, a_nn=a_nn, posratio=posratio)
trainsize = int(len(lot) * 0.75)
train_lot = lot[:trainsize]
test_lot = lot[trainsize:]
print('train data len', len(train_lot))
print('test data len', len(test_lot))
print('q data shape:',train_lot[0][0].shape)
print('a data shape:',train_lot[0][1].shape)
# 負の例を減らす？

# embedIDやめようぜ、学習早くなると思う。



mdl = mimicopynet.model.TransNet3(embed_dim = 1, a_dim = a_dim)

epochnum = 100
t0 = time.time()
for epoch in range(epochnum):
    print('epoch', epoch)
    for data,mode in [(train_lot,'train'), (test_lot,'test')]:
        np.random.shuffle(data)
        bs_normal = 30
        for idx in range(0,len(data),bs_normal):
            batch = data[idx:idx+bs_normal]
            bs = len(batch)
            x = Variable(np.array([b[0] for b in batch]).astype(np.int32))
            t = Variable(np.array([b[1] for b in batch]).astype(np.float32))
            mdl.update(x,t, mode=mode)
            if (time.time() - t0 > 30 or idx+bs_normal>=len(data)):
                t0 = time.time()
                print('mode', mode, '(',idx,'/',len(data),') aveloss', mdl.aveloss(clear=True))
                acc = mdl.getacctable(clear=True)
                precision = acc[1,1]/np.sum(acc[1,:])
                recall = acc[1,1]/np.sum(acc[:,1])
                fvalue = 2*precision*recall/(recall+precision)
                print('acctable:  ', 'P:',precision,'R:',recall,'F:',fvalue)
                print(acc)
            # if (epoch==0 and idx==0 and mode=='train'):
            #     export_graph("computation_graph",[mdl.loss])


# serializers.save_npz('mdl.model', mdl)








