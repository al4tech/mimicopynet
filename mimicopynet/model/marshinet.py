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

class CNN_(chainer.Chain):
    def __init__(self, window=512):
        super(CNN_, self).__init__(
            conv1 = L.Convolution2D(1, 16, ksize=(13,13), pad=(6,6)),
            conv2 = L.Convolution2D(16, 32, ksize=(13,13), pad=(6,6)),
            conv3 = L.Convolution2D(32, 64, ksize=(13,13), pad=(6,6)),
            conv4 = L.Convolution2D(64, 128, ksize=(84,1), pad=(0,0)),
            bn1 = L.BatchNormalization(1),
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
    def __init__(self):
        self.model = L.Classifier(CNN_(), F.sigmoid_cross_entropy, f_measure_accuracy)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
    def load_cqt_inout(self, file):
        data = np.load(file)
        score = data["score"]
        spect = data["spect"]

        width = 128
        length = spect.shape[1]

        spect = [spect[:,i*width:(i+1)*width] for i in range(int(length/width))]
        spect = np.array(spect).astype(np.float32)
        self.spect = np.expand_dims(spect, axis=1)
        score = [score[:,i*width:(i+1)*width] for i in range(int(length/width))]
        self.score = np.array(score).astype(np.int32)
    def eval_call(self, x, t):
        self.model(x, True, t)
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

        trainer.extend(extensions.Evaluator(test_iter, self.model, eval_func=self.eval_call),trigger=(500, 'iteration'))
        trainer.extend(extensions.LogReport(trigger=(50, 'iteration')))
        trainer.extend(extensions.PrintReport(['iteration', 'main/accuracy', 'main/loss', 'validation/main/accuracy', 'validation/main/loss']))
        trainer.extend(extensions.ProgressBar(update_interval=5))
        trainer.extend(extensions.snapshot_object(self.model.predictor,
                                                  'model_{.updater.iteration}.npz',
                                                  serializers.save_npz,
                                                  trigger=(500, 'iteration')))
        trainer.run()
