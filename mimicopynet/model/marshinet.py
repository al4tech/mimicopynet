#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:25:35 2016

@author: marshi
"""

import numpy as np
import scipy as sp
import scipy.io
import librosa
import pretty_midi
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from ..chainer_util import f_measure_accuracy

class CNN_(chainer.Chain):
    def __init__(self, input_cnl=1):
        #TODO: input_cnl mode などはconfigクラスとして一まとめにした方が良いかも
        super(CNN_, self).__init__(
            conv1 = L.Convolution2D(input_cnl, 16, ksize=(13,13), pad=(6,6)),
            conv2 = L.Convolution2D(16, 32, ksize=(13,13), pad=(6,6)),
            conv3 = L.Convolution2D(32, 64, ksize=(13,13), pad=(6,6)),
            conv4 = L.Convolution2D(64, 128, ksize=(84,1), pad=(0,0)),
            bn1 = L.BatchNormalization(input_cnl),
            bn2 = L.BatchNormalization(16),
            bn3 = L.BatchNormalization(32),
            bn4 = L.BatchNormalization(64)
        )

    def __call__(self, x, test=False):
        '''
        x: Variable (bs, 1, pitchs, width)
        ret: Variable (bs, pitchs, width)
        '''
        h = x

        h = self.bn1(h, test=test)
        h = self.conv1(h)
        h = F.leaky_relu(h)

        h = self.bn2(h, test=test)
        h = self.conv2(h)
        h = F.leaky_relu(h)

        h = self.bn3(h, test=test)
        h = self.conv3(h)
        h = F.leaky_relu(h)

        h = self.bn4(h, test=test)
        h = self.conv4(h)

        h = h[:,:,0]
        return h

class CNN(object):
    def __init__(self, input_cnl=1):
        #self.model = L.Classifier(CNN_(), F.sigmoid_cross_entropy, f_measure_accuracy)
        self.model = CNN_(input_cnl=input_cnl)
        self.classifier = L.Classifier(self.model, F.sigmoid_cross_entropy, f_measure_accuracy)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.classifier)
    def load_cqt_inout(self, file):
        '''
        spect np.narray [chl, pitch, seqlen]
        score np.narray [pitch, seqlen]
        '''
        data = np.load(file)
        score = data["score"]
        spect = data["spect"]

        width = 128
        length = spect.shape[2]

        spect = [spect[:,:,i*width:(i+1)*width] for i in range(int(length/width))]
        self.spect = np.array(spect).astype(np.float32)
        score = [score[:,i*width:(i+1)*width] for i in range(int(length/width))]
        self.score = np.array(score).astype(np.int32)
        print("loaded!",self.spect.shape, self.score.shape)
    def eval_call(self, x, t):
        self.classifier(x, True, t)
    def learn(self):
        dataset = chainer.datasets.TupleDataset(self.spect, self.score)
        p = 0.999
        trainn = int(p*len(dataset))
        print(trainn,len(dataset)-trainn)
        train,test = chainer.datasets.split_dataset_random(dataset, trainn)

        train_iter = iterators.SerialIterator(train, batch_size=1, shuffle=True)
        test_iter = iterators.SerialIterator(test, batch_size=2, repeat=False, shuffle=False)

        updater = training.StandardUpdater(train_iter, self.optimizer)
        trainer = training.Trainer(updater, (50000, 'iteration'), out='result')

        trainer.extend(extensions.Evaluator(test_iter, self.classifier, eval_func=self.eval_call),trigger=(500, 'iteration'))
        trainer.extend(extensions.LogReport(trigger=(50, 'iteration')))
        trainer.extend(extensions.PrintReport(['iteration', 'main/accuracy', 'main/loss', 'validation/main/accuracy', 'validation/main/loss']))
        trainer.extend(extensions.ProgressBar(update_interval=5))
        trainer.extend(extensions.snapshot_object(self.model,
                                                  'model_{.updater.iteration}.npz',
                                                  serializers.save_npz,
                                                  trigger=(500, 'iteration')))
        trainer.run()
    def load_model(self, file):
        serializers.load_npz(file, self.model)
    def transcript(self, wavfile, midfile, mode='abs'):

        #TODO: ここの処理はpreprocess.pyに回す
        assert mode=='abs' or mode=='raw'

        data = sp.io.wavfile.read(wavfile)[1]
        data = data.astype(np.float64)
        data = data.mean(axis=1)
        data /= np.abs(data).max()

        if mode == 'abs':
            input_data = np.abs(librosa.core.cqt(data)).astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)
        elif mode == 'raw':
            input_data = librosa.core.cqt(data)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.concatenate([input_data.real, input_data.imag], axis=0).astype(np.float32)

        width = 128
        length = input_data.shape[2]
        input_data = [input_data[:,:,i*width:(i+1)*width] for i in range(int(length/width)+1)]
        input_data = [np.expand_dims(input_data_, axis=0) for input_data_ in input_data]

        score = []
        for input_data_ in input_data:
            score_ = (self.model(input_data_, test=True)[0].data>0.)*1
            score.append(score_)

        score = np.concatenate(score, axis=1)

        pitch, time = np.where(score==1)
        dif = [time[i+1]-time[i] for i in range(len(pitch)-1)]
        end = np.concatenate([np.where(np.array(dif)!=1)[0],[len(time)-1]])
        start = np.concatenate([[0],end[:-1]+1])
        pitch = pitch[start]
        end = time[end]
        start = time[start]

        hz = 44100
        tempo = 120
        res = 960
        pm = pretty_midi.PrettyMIDI(resolution=res, initial_tempo=tempo)
        instrument = pretty_midi.Instrument(0)

        sample_t = (512/hz)
        for s_,e_,p_ in zip(start, end, pitch):
            note = pretty_midi.Note(velocity=100, pitch=p_, start=s_*sample_t, end=e_*sample_t)
            instrument.notes.append(note)
        pm.instruments.append(instrument)
        pm.write(midfile)
