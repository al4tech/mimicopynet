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

from chainer import reporter

class FMeasureAccuracy(chainer.function.Function):

    ignore_label = -1

    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        chainer.utils.type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == np.int32,
            t_type.shape == x_type.shape,
        )

    def forward(self, inputs):
        xp = chainer.cuda.get_array_module(*inputs)
        y, t = inputs
        # flatten
        y = y.ravel()
        t = t.ravel()
        c = (y >= 0)

        tp = ((c==1)*(t==1)).sum()
        fp = ((c==1)*(t==0)).sum()
        fn = ((c==0)*(t==1)).sum()
        rec = tp/(tp+fn)
        pre = tp/(tp+fp)
        # reporter.report({'precision': pre, 'recall': rec}, self) # これで良い？
        return xp.asarray(2*rec*pre/(rec+pre), dtype=y.dtype),


def f_measure_accuracy(y, t):
    """
    F.binary_accuracyのF値バージョン
    """
    return FMeasureAccuracy()(y, t)
