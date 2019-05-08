import numpy as np
import sys
from chainer import Chain, ChainList, cuda, gradient_check, Function, Link, \
    optimizers, serializers, utils, Variable, datasets, using_config, training, iterators
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L

class Model(object):
    def __init__(self):
        pass


    def fit(self, wsh5s, eval_wsh5s=None):
        """

        Args:
            wsh5s (iterable of str):
                wsh5 形式のファイルへのパスを各要素に持つ iterable.
        """
        pass


    def predict_proba(self, X):
        pass
