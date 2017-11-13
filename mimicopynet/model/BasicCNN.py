#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:25:35 2016

@author: marshi
"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from ..chainer_util import f_measure_accuracy
from ..data import make_cqt_input, score_to_midi, score_to_image

class BasicCNN_(chainer.Chain):
    '''
    下のBasicCNNで使うChain
    '''
    def __init__(self, input_cnl=1):
        '''
        input_cnl: 画像のチャンネル（CQTの絶対値を使うなら1,実部と虚部を使うなら2)
        '''
        #TODO: input_cnl mode などはconfigクラスとして一まとめにした方が良いかも
        super(BasicCNN_, self).__init__(
            conv1 = L.Convolution2D(input_cnl, 4, ksize=(13,13), pad=(6,6)),
            conv2 = L.Convolution2D(4, 8, ksize=(13,13), pad=(6,6)),
            conv3 = L.Convolution2D(8, 16, ksize=(13,13), pad=(6,6)),
            conv4 = L.Convolution2D(16, 128, ksize=(84,1), pad=(0,0)),
            bn1 = L.BatchNormalization(input_cnl),
            bn2 = L.BatchNormalization(4),
            bn3 = L.BatchNormalization(8),
            bn4 = L.BatchNormalization(16)
        )
        self.cnt_call = 0

    def __call__(self, x, test=False):
        '''
        x: Variable (bs, 1, pitchs, width)

        ret: Variable (bs, pitchs, width)
            sigmoidをかけない値を出力する．
        '''
        assert(isinstance(test, bool))
        with chainer.using_config('train', not test):

            # (bs, cnl, 84, 128)
            h = x

            h = self.bn1(h)#, test=test)
            h = self.conv1(h)
            h = F.leaky_relu(h)

            # (bs, 4, 84, 128)

            h = self.bn2(h)#, test=test)
            h = self.conv2(h)
            h = F.leaky_relu(h)

            # (bs, 8, 84, 128)

            h = self.bn3(h)#, test=test)
            h = self.conv3(h)
            h = F.leaky_relu(h)

            # (bs, 16, 84, 128)

            h = self.bn4(h)#, test=test)
            h = self.conv4(h)

            # (bs, 128, 1, 128)
            h = h[:,:,0]

            # (bs, 128, 128)
            self.cnt_call += 1            
        return h

class MarioCNN_(chainer.Chain):
    '''
    試しに作ってみたChain
    '''
    def __init__(self, input_cnl=1, skip=False):
        '''
        input_cnl: 画像のチャンネル（CQTの絶対値を使うなら1,実部と虚部を使うなら2)
        skip: スキップ結合の有無(bool)
        '''
        #TODO: input_cnl mode などはconfigクラスとして一まとめにした方が良いかも
        super(MarioCNN_, self).__init__(
            conv1 = L.Convolution2D(input_cnl, 4, ksize=(49,3), pad=(24,1), stride=(1,2)),
            conv2 = L.Convolution2D(4, 8, ksize=(49,3), pad=(24,1), stride=(1,2)),
            conv3 = L.Convolution2D(8, 16, ksize=(3,3), pad=(1,1), stride=(1,2)),
            conv4 = L.Convolution2D(16, 128, ksize=(84,16), pad=(0,0)),
            deconv1 = L.Deconvolution2D(128, 16, ksize=(128,16), pad=(0,0)),
            deconv2 = L.Deconvolution2D(16, 8, ksize=(3,3), pad=(1,1), stride=(1,2), outsize=(128,32)),
            deconv3 = L.Deconvolution2D(8, 4, ksize=(49,3), pad=(24,1), stride=(1,2), outsize=(128,64)),
            deconv4 = L.Deconvolution2D(4, 1, ksize=(49,3), pad=(24,1), stride=(1,2), outsize=(128,128)),
            bn1 = L.BatchNormalization(input_cnl),
            bn2 = L.BatchNormalization(4),
            bn3 = L.BatchNormalization(8),
            bn4 = L.BatchNormalization(16),
            bn5 = L.BatchNormalization(128),
            bn6 = L.BatchNormalization(16),
            bn7 = L.BatchNormalization(8),
            bn8 = L.BatchNormalization(4)
        )
        self.skip = skip
        self.cnt_call = 0

    def __call__(self, x, test=False):
        '''
        x: Variable (bs, 1, pitchs, width)

        ret: Variable (bs, pitchs, width)
            sigmoidをかけない値を出力する．
        '''
        assert(isinstance(test, bool))
        with chainer.using_config('train', not test):

            # (bs, cnl, 84, 128)
            h = x

            if self.skip:
                bs = h.shape[0]
                zero_bs_4_24_64 = Variable(np.zeros([bs, 4, 24, 64]).astype(np.float32))
                zero_bs_4_20_64 = Variable(np.zeros([bs, 4, 20, 64]).astype(np.float32))
                zero_bs_8_24_32 = Variable(np.zeros([bs, 8, 24, 32]).astype(np.float32))
                zero_bs_8_20_32 = Variable(np.zeros([bs, 8, 20, 32]).astype(np.float32))
                zero_bs_16_24_16 = Variable(np.zeros([bs, 16, 24, 16]).astype(np.float32))
                zero_bs_16_20_16 = Variable(np.zeros([bs, 16, 20, 16]).astype(np.float32))


            h = self.bn1(h)
            h = self.conv1(h)
            h = F.relu(h)
            assert(h.data.shape[1:] == (4, 84, 64)), h.data.shape
            if self.skip: s1 = F.concat((zero_bs_4_24_64, h, zero_bs_4_20_64), axis=2)

            h = self.bn2(h)
            h = self.conv2(h)
            h = F.relu(h)
            assert(h.data.shape[1:] == (8, 84, 32)), h.data.shape
            if self.skip: s2 = F.concat((zero_bs_8_24_32, h, zero_bs_8_20_32), axis=2)

            h = self.bn3(h)
            h = self.conv3(h)
            h = F.relu(h)
            assert(h.data.shape[1:] == (16, 84, 16)), h.data.shape
            if self.skip: s3 = F.concat((zero_bs_16_24_16, h, zero_bs_16_20_16), axis=2)

            h = self.bn4(h)
            h = self.conv4(h)
            h = F.relu(h)
            assert(h.data.shape[1:] == (128, 1, 1)), h.data.shape

            h = self.bn5(h)
            h = self.deconv1(h)
            h = F.relu(h)
            assert(h.data.shape[1:] == (16, 128, 16)), h.data.shape
            if self.skip: h += s3

            h = self.bn6(h)
            h = self.deconv2(h)
            h = F.relu(h)
            assert(h.data.shape[1:] == (8, 128, 32)), h.data.shape
            if self.skip: h += s2

            h = self.bn7(h)
            h = self.deconv3(h)
            h = F.relu(h)
            assert(h.data.shape[1:] == (4, 128, 64)), h.data.shape
            if self.skip: h += s1

            h = self.bn8(h)
            h = self.deconv4(h)
            assert(h.data.shape[1:] == (1, 128, 128)), h.data.shape
            # (bs, 1, 128, 128)

            h = h[:,0,:,:]

            # (bs, 128, 128)
            self.cnt_call += 1            
        return h

class BasicCNN(object):
    '''
    スペクトル×時間の２次元画像から，それぞれの時間における耳コピを行うモデル
    self.model: BasicCNN_のインスタンス
    self.classifier: 分類するためのクラス
    '''
    def __init__(self, input_cnl=1, gpu=None):
        '''
        input_cnl: 画像のチャンネル（CQTの絶対値を使うなら1,実部と虚部を使うなら2)
        '''
        # self.model = BasicCNN_(input_cnl=input_cnl)
        self.model = MarioCNN_(input_cnl=input_cnl, skip=True)
        if gpu is not None:
            cuda.get_device(gpu).use()
            self.model.to_gpu(gpu)
            self.xp = cuda.cupy
        else:
            self.xp = np
        self.classifier = L.Classifier(self.model, F.sigmoid_cross_entropy,
                                       f_measure_accuracy)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.classifier)
    def load_cqt_inout(self, file):
        '''
        spect np.narray [chl, pitch, seqlen]
        score np.narray [pitch, seqlen]
        '''
        data = np.load(file)
        score = self.xp.array(data["score"], dtype=self.xp.int32)
        spect = self.xp.array(data["spect"], dtype=self.xp.float32)
        del data

        width = 128
        stride = 32
        length = spect.shape[2]

        spect = [spect[:,:,i*stride:i*stride+width] for i in range((length-width)//stride)]
        self.spect = spect # xp.array(spect)
        score = [score[:,i*stride:i*stride+width] for i in range((length-width)//stride)]
        self.score = score # xp.array(score)
        print("Loaded!")
        print(' length of original data (in .npz):', length)
        print(' number of data (== len(self.spect) == len(self.score)):', len(self.spect))
        print(' shape of each spect data:', self.spect[0].shape, self.spect[0].dtype)
        print(' shape of each score data:', self.score[0].shape, self.score[0].dtype)
    def eval_call(self, x, t):
        '''
        テスト用にClassifierを呼ぶ
        self.classifier(x, t)と同様の使い方をする．

        x: (bs, cnl, pitch_in, width)
        t: (bs, pitch_out, width)
        '''
        self.classifier(x, True, t) #これはlossを返す。TODO: Trueは何？
        
        # self.classfier.y: (bs, pitch_out, width)

    def learn(self, iter_num=300000):
        '''
        学習をするメソッド
        '''
        dataset = chainer.datasets.TupleDataset(self.spect, self.score)
        p = 0.9
        trainn = int(p*len(dataset))
        print(trainn,len(dataset)-trainn)
        train,test = chainer.datasets.split_dataset_random(dataset, trainn)

        train_iter = iterators.SerialIterator(train, batch_size=10, shuffle=True)
        test_iter = iterators.SerialIterator(test, batch_size=10, repeat=False,
                                             shuffle=False)

        updater = training.StandardUpdater(train_iter, self.optimizer)
        trainer = training.Trainer(updater, (iter_num, 'iteration'), out='result')

        trainer.extend(extensions.Evaluator(test_iter, self.classifier,
                                            eval_func=self.eval_call),
                                            trigger=(500, 'iteration'))
        trainer.extend(extensions.LogReport(trigger=(500, 'iteration')))
        trainer.extend(extensions.PrintReport(['iteration', 'main/accuracy',
                                               'main/loss',
                                               'validation/main/accuracy',
                                               'validation/main/loss']))
        trainer.extend(extensions.ProgressBar(update_interval=5))
        trainer.extend(extensions.snapshot_object(self.model,
                                            'model_{.updater.iteration}.npz',
                                            serializers.save_npz),
                                            trigger=(100000, 'iteration'))
        trainer.run()
    def load_model(self, file):
        serializers.load_npz(file, self.model)
    def __call__(self, input_data):
        '''
        学習したモデルを動かします
        !! 返り値を pre-sigmoid に変更しました(以前は pre-sigmoid を step-function したもの(0 or 1)を返していた)
        TODO: gpu対応？

        input: np.narray [chl, pitch_in, seqlen]
        ret:  np.narray [pitch_out, seqlen]
        '''
        pitch_out = 128

        ret = np.zeros([pitch_out, input_data.shape[2]]).astype(np.float)
        ret_denom = np.zeros([pitch_out, input_data.shape[2]]).astype(np.int)

        width = 128
        stride = 8
        length = input_data.shape[2]

        data_list, pos_list = [], []
        for offset in range(0, width, stride):
            data_list += [np.expand_dims(input_data[:,:,offset+i*width:offset+(i+1)*width], axis=0)
                          for i in range((length-offset)//width)] # each element: (1, cnl, 84, 128)
            pos_list += [offset+i*width for i in range((length-offset)//width)]

        bs = 1
        score = []
        for i in range(0, len(data_list), bs):
            minibatch = np.concatenate(data_list[i:i+bs], axis=0)
            # score_ = (self.model(minibatch, test=True).data>0.)*1
            # # self.model は pre-sigmoid な値を出力するので、閾値は 0 で OK
            
            # score_ = self.model(minibatch, test=True).data # (bs, pitch_out, width)
            # score += list(score_)

            score = self.model(minibatch, test=True).data # (bs, pitch_out, width)
            for j in range(bs):
                offset = pos_list[i+j]
                ret[:,offset:offset+width] += score[j,:,:]
                ret_denom[:,offset:offset+width] += 1
        # output = np.concatenate(score, axis=1)
        output = ret / np.maximum(ret_denom, 1)
        return output
    def transcript(self, wavfile, midfile, imgfile=None, mode='abs'):
        '''
        学習したモデルを使って，耳コピをするメソッド
        wavfile: 耳コピしたいWavファイルのファイル名(44100,2ch想定)
        midfile: 耳コピして生成される，midファイル名
        model: CQTからどんな値を抽出するか
            'abs' 絶対値(chl=1)
            'raw' 実部と虚部をそのままだす(chl=2)
        '''
        input_data = make_cqt_input(wavfile, mode=mode)
        pre_sigmoid_score = self(input_data)
        digital_score = (pre_sigmoid_score > 0.) * 1 # 0と1
        sigmoided_score = 1. / (1. + np.exp(-pre_sigmoid_score)) # 0以上1以下

        score_to_midi(digital_score, midfile)
        if imgfile is not None:
            score_to_image(sigmoided_score, imgfile)
