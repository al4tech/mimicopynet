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
import glob


def export_graph(filename, vs):
    import chainer.computational_graph as ccg
    g = ccg.build_computational_graph(vs)
    with open(filename, 'w') as o:
        o.write(g.dump())

def in_npy_out_npy_to_lot(in_npy, out_npy, q_sample=5120, a_sample=10, a_nn=list(range(128)), posratio=None, shuffle=True):
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
    if (shuffle):
        # 最後にシャッフル
        np.random.shuffle(qa) # list_of_tuple形式は順不同とする (TODO: この点で、set_of_tupleの方が妥当？)
    return qa


def mndata_to_lot(mndata, q_sample=5120, a_sample=10, a_nn=list(range(128)), posratio=None, stride=1, shuffle=True):
    '''
    list_of_tuple形式に変換します
    mndata : mnnpzファイル(.npz)をnp.loadして得られるオブジェクト
    a_nn <list of int> : ピアノロールのうち教師データとするノートナンバーのリスト（1音高だけ耳コピさせたい時は [60] などと指定）
    posratio <None or float> : 教師データが0ベクトルではないようなサンプルが全体のうちこの割合になるように調整する(どうやって？)
    stride <int> : mndataに入ってるピアノロールのサンプル点から、いくつおきにlotへ抽出するか（デフォルト：1　すなわち全部）

    各 i について、
    ピアノロールデータ score　の　[i : i+a_sample] サンプル目　と、
    波形データ wave の round( score_sample[i + (a_sample-1)/2] - (q_sample-1)/2)サンプル目から連続するq_sampleサンプルと対応させます。
    
    基本的には a_sample==1 推奨です。(mndataでは score_sampleが時間順に並んでいる保証はないので)

    returns: list_of_tuple形式 各qはnp.array (r), 各aはnp.array (128)
    '''
    if (a_sample != 1):
        print('Warning: a_sample != 1')
    wave, score, score_sample = mndata['wave'], mndata['score'], mndata['score_sample']
    s = wave.shape[0]
    n = score.shape[1]
    qa = []
    padded_wave = np.array([wave[0]] * q_sample + list(wave) + [wave[-1]] * q_sample)
    for i in range(0, n - a_sample + 1, stride):
        # start = q_sample + int(np.round(s/n * (i+(a_sample-1)/2) - (q_sample-1)/2))
        start = q_sample + int(score_sample[int(np.round(i+(a_sample-1)/2))] - (q_sample-1)/2)
        qa.append((padded_wave[start:start+q_sample], score[a_nn,i:i+a_sample]))

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
    if (shuffle):
        # 最後にシャッフル
        np.random.shuffle(qa) # list_of_tuple形式は順不同とする (TODO: この点で、set_of_tupleの方が妥当？)
    return qa



def test_on_mid(model, mid_file):
    '''
    midiファイルをどれくらい耳コピできるかを調べます
    model <chainer.Chain>: エージェントのモデル。というか学習済みのTransNet3()。update(x,t,mode)が存在している必要あり。
    mid_file <string>: テストしたいmidiファイル　（同名の.wavファイルが（存在しない場合）作られます。）
    a_nn <list of int>: テストするノートナンバーのリスト
    TODO: ノートナンバーごとに正答率見たい？そうでもない？ (現状、集計をTransNet3に任せてるのですぐ対応は難しいかも）
    →むしろTransNet3にノートナンバーごと正答率を求める機能をつければ良いのでは？
    '''
    data_in, data_out = mid_to_in_npy_out_npy(mid_file)
    lot = in_npy_out_npy_to_lot(data_in, data_out,
                                q_sample=model.fmdl.ssize*model.fmdl.slen, # 10 * 512 # 1 * 5120
                                a_sample=model.fmdl.ssize*model.fmdl.a_dim, # 10 * 1 # 1 * 1
                                a_nn=model.a_nn, shuffle=False)
    # ▲ in_npyではmu-law変換済みになっている。他方、mnnpz形式では生波形のままになっている。注意！
    t0 = time.time()
    print('start testing.....................(',mid_file,')')
    bs_normal = 30
    mode = 'test'
    data = lot
    o_roll = [] # output roll
    a_roll = [] # answer roll
    for idx in range(0,len(data),bs_normal):
        batch = data[idx:idx+bs_normal]
        bs = len(batch)
        x = Variable(np.array([b[0] for b in batch]).astype(np.int32))
        t = Variable(np.array([b[1] for b in batch]).astype(np.float32))
        model.update(x,t, mode=mode)
        o_roll += list(model.lastoutput.transpose(1,0,2).reshape(len(model.a_nn), -1).T) # リストの各要素は　長さ a_dim のリストである
        a_roll += list(model.lastanswer.transpose(1,0,2).reshape(len(model.a_nn), -1).T) # リストの各要素は　長さ a_dim のリストである
        if (time.time() - t0 > 600 or idx+bs_normal>=len(data)):
            t0 = time.time()
            print('mode', mode, '(',idx,'/',len(data),') aveloss', model.aveloss(clear=False))
            acc = model.getacctable(clear=(idx+bs_normal>=len(data)))
            precision = acc[1,1]/np.sum(acc[1,:])
            recall = acc[1,1]/np.sum(acc[:,1])
            fvalue = 2*precision*recall/(recall+precision)
            print('acctable:  ', 'P:',precision,'R:',recall,'F:',fvalue)
            print(acc)
    print('done testing.')
    import matplotlib.pyplot as plt
    o_roll = np.array(o_roll).T
    a_roll = np.array(a_roll).T
    idx = np.linspace(0, o_roll.shape[1], 100).astype(np.int32)[:-1]
    plt.subplot(211)
    plt.imshow((o_roll[:,idx] > 0.5).astype(np.int32))
    plt.subplot(212)
    plt.imshow(a_roll[:,idx])
    plt.show()


def test_on_mnnpz(model, mnnpz_file):
    '''
    mnnpzファイルをどれくらい耳コピできるかを調べます
    model <chainer.Chain>: エージェントのモデル。というか学習済みのTransNet3()。update(x,t,mode)が存在している必要あり。
    mnnpz_file <string>: テストしたいmnnpzファイル
    a_nn <list of int>: テストするノートナンバーのリスト
    TODO: ノートナンバーごとに正答率見たい？そうでもない？ (現状、集計をTransNet3に任せてるのですぐ対応は難しいかも）
    →むしろTransNet3にノートナンバーごと正答率を求める機能をつければ良いのでは？
    '''
    lot = mnnpz_to_lot(mnnpz_file, q_sample=model.fmdl.ssize*model.fmdl.slen, # 10 * 512 # 1 * 5120
                                a_sample=model.fmdl.ssize*model.fmdl.a_dim, # 10 * 1 # 1 * 1
                                a_nn=model.a_nn, shuffle=False)
    t0 = time.time()
    print('start testing.....................(',mnnpz_file,')')
    bs_normal = 30
    mode = 'test'
    data = lot
    o_roll = [] # output roll
    a_roll = [] # answer roll
    for idx in range(0,len(data),bs_normal):
        batch = data[idx:idx+bs_normal]
        bs = len(batch)
        x = Variable(np.array([b[0] for b in batch]).astype(np.int32))
        t = Variable(np.array([b[1] for b in batch]).astype(np.float32))
        model.update(x,t, mode=mode)
        o_roll += list(model.lastoutput.transpose(1,0,2).reshape(len(model.a_nn), -1).T) # リストの各要素は　長さ a_dim のリストである
        a_roll += list(model.lastanswer.transpose(1,0,2).reshape(len(model.a_nn), -1).T) # リストの各要素は　長さ a_dim のリストである
        if (time.time() - t0 > 600 or idx+bs_normal>=len(data)):
            t0 = time.time()
            print('mode', mode, '(',idx,'/',len(data),') aveloss', model.aveloss(clear=False))
            acc = model.getacctable(clear=(idx+bs_normal>=len(data)))
            precision = acc[1,1]/np.sum(acc[1,:])
            recall = acc[1,1]/np.sum(acc[:,1])
            fvalue = 2*precision*recall/(recall+precision)
            print('acctable:  ', 'P:',precision,'R:',recall,'F:',fvalue)
            print(acc)
    print('done testing.')
    import matplotlib.pyplot as plt
    o_roll = np.array(o_roll).T
    a_roll = np.array(a_roll).T
    idx = np.linspace(0, o_roll.shape[1], 100).astype(np.int32)[:-1]
    plt.subplot(211)
    plt.imshow((o_roll[:,idx] > 0.5).astype(np.int32))
    plt.subplot(212)
    plt.imshow(a_roll[:,idx])
    plt.show()


def mid_to_in_npy_out_npy(mid_file):
    '''
    midファイルから、in_npyとout_npyの組を返します。

    具体的には
    .midから.wavを生む
    .midと.wavから.mid.npyと.wav.npyを生む
    .mid.npyと.wav.npyからin_npyとout_npyを読み込み、返す
    ということをやります

    すでに生成済みのファイルは再生成しません。
    '''
    import os.path
    base, ext = os.path.splitext(mid_file)
    assert(ext==".mid")
    wav_file = base+".wav"
    midnpy_file = base+".mid.npy"
    wavnpy_file = base+".wav.npy"

    if not(os.path.exists(wavnpy_file)):
        if not(os.path.exists(wav_file)):
            mimicopynet.data.midi_to_wav(mid_file, wav_file)
        mimicopynet.data.wav_to_input(wav_file, wavnpy_file)

    if not(os.path.exists(midnpy_file)):
        mimicopynet.data.midi_to_output(mid_file, midnpy_file)

    return np.load(wavnpy_file), np.load(midnpy_file)


def mid_to_lot(mid_file, q_sample=5120, a_sample=10, a_nn=list(range(128)), posratio=None, samplenum=None, shuffle=True):
    '''
    mid_to_in_npy_out_npy → in_npy_out_npy_to_lot
    mid_file <str>: midファイル名　ただしglob.globに渡せるワイルドカード表現も可能。該当するファイル全てからデータを取得する。
    samplenum <None or int>: サンプル数の上限。
    '''
    files = sorted(glob.glob(mid_file))
    print("files matched:", len(files))
    lot = []
    for i,file in enumerate(files):
        print(i,"/",len(files),":",file,'(num of samples collected:',len(lot),')')
        in_npy, out_npy = mid_to_in_npy_out_npy(file)
        lot += in_npy_out_npy_to_lot(in_npy, out_npy, q_sample=q_sample, a_sample=a_sample, a_nn=a_nn, posratio=posratio, shuffle=shuffle)
        if (samplenum is not None and samplenum <= len(lot)):
            lot = lot[:samplenum]
            break
    if (shuffle):
        # 最後にシャッフル
        np.random.shuffle(lot)
    return lot


def mnnpz_to_lot(mnnpz_file, q_sample=5120, a_sample=1, a_nn=list(range(128)), posratio=None, stride=1, samplenum=None, shuffle=True):
    '''
    mnnpz_file <str>: mnnpzファイル名　ただしglob.globに渡せるワイルドカード表現も可能。該当するファイル全てからデータを取得する。
    samplenum <None or int>: lotに含まれるサンプル数の上限。
    '''
    import glob
    files = sorted(glob.glob(mnnpz_file))
    print("files matched:", len(files))
    lot = []
    for i,file in enumerate(files):
        print(i,"/",len(files),":",file,'(num of samples collected:',len(lot),')')
        data = np.load(file)
        lot += mndata_to_lot(data, q_sample=q_sample, a_sample=a_sample, a_nn=a_nn, posratio=posratio, stride=stride, shuffle=shuffle)
        if (samplenum is not None and samplenum <= len(lot)):
            lot = lot[:samplenum]
            break
    if (shuffle):
        # 最後にシャッフル
        np.random.shuffle(lot)
    return lot    


'''
def harmonic_score(n):
    """
    ノートナンバー n との調和度の高さ（同時に鳴った時にnが鳴ってるかの判定が難しくなる度合い）を
    各ノートナンバーについて求めて、長さ128のarrayで返します。
    (make_random_songの引数lmb_startとlmb_stopに与える用)
    """
    harmonic_dnn = [12, 19, 24, 28, 31, 34, 36, 38, 40]
    harmonic_dnn = [-i for i in harmonic_dnn][::-1] + [0] + harmonic_dnn
    ret = np.ones(128).astype(np.float32) # 1 が標準としよう。
    for i in harmonic_dnn:
        if (0 <= n+i < 128): ret[n+i] = 50.0 # 5はすごい
    return ret
lmb_start = list(50. / harmonic_score(60))
lmb_stop = 1.
mimicopynet.data.make_random_song("hard.mid", lmb_start=lmb_start, lmb_stop=lmb_stop)
mimicopynet.data.midi_to_wav("hard.mid","hard.wav")
mimicopynet.data.midi_to_output("hard.mid","hard.mid.npy")
mimicopynet.data.wav_to_input("hard.wav","hard.wav.npy")
'''

'''
mimicopynet.data.make_random_song("train.mid")
mimicopynet.data.midi_to_wav("train.mid","train.wav")
mimicopynet.data.midi_to_output("train.mid","train_out.npy")
mimicopynet.data.wav_to_input("train.wav","train_in.npy")
'''

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


　・random songはマシンにとってはとても簡単であることが判明したので実際のソングにしよう

・過学習がひどい（train F: 0.94、valid F: 0.56、test F: 0.28）。q_sampleを512から増やした方が良いか。
slenとssizeをが512と10だったが5120と1にする
yosnetの中身は、512-512-256-1 だったのを 5120-1280-320-1にする
batchsizeは30だったのを300にする（高速化のため。5倍くらい早くなった）

相変わらず過学習がひどい（train F: 0.99(40epoch), valid F: 0.77(2epoch)）
次元じゃなくてサンプル数増やさないとダメかも。今は約16万サンプル
これを50万サンプルまで増やそう




・オンラインNMF？



midiworldで、正常にwav変換できなかったファイルたち
A_Teens/Super_Trouper.mid

midiworldで、正常にmid.npy変換できなかったファイルたち
Aaron_Neville/Tell_It_Like_It_Is.mid
'''







def mulaw(wav, mu=255):
    wav = np.sign(wav)*np.log(1+mu*np.abs(wav))/np.log(1+mu)
    # wav = np.round((wav + 1) / 2 * mu).astype(np.int32)  
    wav = (wav + 1.) / 2 * mu
    return wav  

def get_train_test_lot(trainpath, trainsample, testpath, testsample, scaling='linear'):
    '''
    trainpath = '/Users/yoshidayuuki/Downloads/musicnet/musicnet_data/2*/data.npz'
    trainsample = 375000
    testpath = '/Users/yoshidayuuki/Downloads/musicnet/musicnet_data/1*/data.npz'
    testsample = 125000
    '''
    lot = mnnpz_to_lot(trainpath,
        q_sample=slen*ssize, a_sample=ssize, a_nn=a_nn, posratio=posratio, stride=2, samplenum=trainsample)
    # raw waveなのを 0〜255の整数階調に変換する（あとで元に戻すから二度手間なんだけど・・・）
    lot_new = []
    for tup in lot:
        # from: -1〜1 float / to: 0〜255の整数値
        assert(min(tup[0])>=-1. and max(tup[0])<=1.)
        if scaling=='linear':
            lot_new.append(((tup[0]+1.)*256, tup[1])) # 線形変換 (TODO: mu-law)
        elif scaling=='mulaw':
            lot_new.append((mulaw(tup[0]), tup[1])) # mu-law
        else:
            raise ValueError
    train_lot = lot_new

    lot = mnnpz_to_lot(testpath,
        q_sample=slen*ssize, a_sample=ssize, a_nn=a_nn, posratio=posratio, stride=2, samplenum=testsample)
    # raw waveなのを 0〜255の整数階調に変換する（あとで元に戻すから二度手間なんだけど・・・）
    lot_new = []
    for tup in lot:
        # from: -1〜1 float / to: 0〜255の整数値
        assert(min(tup[0])>=-1. and max(tup[0])<=1.)
        if scaling=='linear':
            lot_new.append(((tup[0]+1.)*256, tup[1])) # 線形変換 (TODO: mu-law)
        elif scaling=='mulaw':
            lot_new.append((mulaw(tup[0]), tup[1])) # mu-law
        else:
            raise ValueError
    test_lot = lot_new
    print('train data len', len(train_lot))
    print('test data len', len(test_lot))
    print('q data shape:',train_lot[0][0].shape)
    print('a data shape:',train_lot[0][1].shape)
    return train_lot, test_lot


a_nn = [60] # list(np.arange(60, 72)) # 1音に限定した耳コピはどうだろう
a_dim = len(a_nn)
posratio = None # 正例の割合
slen = 5120
ssize = 1

np.random.seed(0)


mdl = mimicopynet.model.TransNet3(fmdl=mimicopynet.model.yosnet(a_dim=a_dim, slen=slen, ssize=ssize), a_nn=a_nn)
# TransNet3: エージェント
# yosnet, wavenet: 脳


mdlfilename = "fmdl60.model"
try:
    #hogehoge
    serializers.load_npz(mdlfilename, mdl.fmdl)
    print('loaded.')
    hogehoge
except: 
    print('welcome to training!')
#    train_lot, test_lot = get_train_test_lot('/Users/yoshidayuuki/Downloads/musicnet/musicnet_data/2*/data.npz', 375000,
#                                            '/Users/yoshidayuuki/Downloads/musicnet/musicnet_data/1*/data.npz', 125000)
    train_lot, test_lot = get_train_test_lot('/Users/yoshidayuuki/Downloads/musicnet/musicnet_data/2*/data.npz', 1,
                                            '/Users/yoshidayuuki/Downloads/musicnet/musicnet_data/1727/data.npz', None)
    epochnum = 100
    fmax = np.zeros(2)
    t0 = time.time()
    for epoch in range(epochnum):
        print('epoch', epoch)
        for data,mode in [(train_lot,'train'), (test_lot,'test')]:
            np.random.shuffle(data) # ここでもシャッフル
            bs_normal = 300
            for idx in range(0,len(data),bs_normal):
                batch = data[idx:idx+bs_normal]
                bs = len(batch)
                x = Variable(np.array([b[0] for b in batch]).astype(np.int32))
                t = Variable(np.array([b[1] for b in batch]).astype(np.float32))
                mdl.update(x,t, mode=mode)
                if (time.time() - t0 > 600 or idx+bs_normal>=len(data)):
                    t0 = time.time()
                    endflg = (idx+bs_normal>=len(data))
                    print('mode', mode, '(',idx,'/',len(data),') aveloss', mdl.aveloss(clear=endflg))
                    acc = mdl.getacctable(clear=endflg)
                    precision = acc[1,1]/np.sum(acc[1,:])
                    recall = acc[1,1]/np.sum(acc[:,1])
                    fvalue = 2*precision*recall/(recall+precision)
                    fmax[mode=='test'] = max(fmax[mode=='test'], fvalue)
                    print('acctable:  ', 'P:',precision,'R:',recall,'F:',fvalue,'(Fmax:',fmax[mode=='test'],')')
                    print(acc)
                    if (mode=='test' and fmax[1] == fvalue):
                        pass
                        # serializers.save_npz(mdlfilename, mdl.fmdl)
                        # print('saved: '+mdlfilename)
                # if (epoch==0 and idx==0 and mode=='train'):
                #     export_graph("computation_graph",[mdl.loss])
    # serializers.save_npz(mdlfilename, mdl.fmdl)
# test_on_mid(mdl, "yos.mid")
for f in glob.glob("/Users/yoshidayuuki/Downloads/musicnet/musicnet_data/1*/data.npz"):
    test_on_mnnpz(mdl, f)






