import numpy as np
import sys
from chainer import Chain, ChainList, cuda, gradient_check, Function, Link, \
    optimizers, serializers, utils, Variable, datasets, using_config, training, iterators
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L

from ..data import WSH5Dataset
from ..chainer_util import f_measure_accuracy

class Net(Chain):
    def __init__(self):
        super(self, Net).__init__(
            l1=L.Linear(None, 128)
        )
    def __call__(self, X):
        """
        Args:
            X (Variable):
                音声波形．shape==(*, samples_of_wave,) dtype==xp.float32
        Returns:
            y (Variable):
                「各音がその間に一度でも新規に鳴ったかどうか」の判定結果 (pre_sigmoid value)
                 shape==(*, 128) dtype==xp.float32
        """
        y = self.l1(X)
        return y

    @staticmethod
    def lossfun(y, t):
        """
        Args:
            y (Variable) 上参照
            t (Variable) shape==(*, 2, 128, samples_of_score) dtype==xp.int32
        """
        return F.sigmoid_cross_entropy(y, t[:,1,:,:].max(axis=-1))

    @staticmethod
    def accfun(y, t):
        """
        Args:
            y (Variable) 上参照
            t (Variable) shape==(*, 2, 128, samples_of_score) dtype==xp.int32
        """
        return f_measure_accuracy(y, t[:,1,:,:].max(axis=-1))


class Model(object):
    def __init__(self):
        pass


    def fit(self, wsh5s, eval_wsh5s=None, **kwargs):
        """
        Args:
            wsh5s (iterable of str):
                wsh5 形式のファイルへのパスを各要素に持つ iterable.
        """
        bs_train = 100
        bs_eval = 100
        gpu = -1
        result_dir = 'result'
        num_epoch = 20
        train_size = 10000
        eval_size = 1000

        # 入出力のサイズを確定させる．（使用しないかも．）
        length_sec = 0.5
        ws = WSH5(wsh5s[0])
        samples_of_wave, samples_of_score = ws.get_sample_length(length_sec)

        # データセットの準備．
        dataset_train = WSH5Dataset(wsh5s, train_size, length_sec=length_sec, no_drum=True, num_part=1)
        if eval_wsh5s is not None:
           dataset_eval = WSH5Dataset(eval_wsh5s, eval_size, length_sec=length_sec, no_drum=True, num_part=1)

        # Chain の準備
        net = Net()
        mdl = L.Classifier(net, lossfun=net.lossfun, accfun=net.accfun)
        opt = optimizers.Adam().setup(mdl)
        itr_train = iterators.SerialIterator(dataset_train, shuffle=False, batch_size=bs_train)
        upd = training.StandardUpdater(itr_train, opt, device=gpu)
        trn = training.Trainer(upd, (num_epoch, 'epoch'), out=logdir)
        trn.extend(extensions.LogReport())
        trn.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
        trn.extend(extensions.snapshot_object(mdl, filename='model_epoch-{.updater.epoch}'))
        if eval_wsh5s is not None:
            itr_eval = iterators.SerialIterator(dataset_eval, shuffle=False, repeat=False, batch_size=bs_eval)
            trn.extend(extensions.Evaluator(itr_eval, mdl, device=gpu))
        trn.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'main/precision', 'main/recall', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
        trn.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
        trn.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
        trn.extend(extensions.dump_graph('main/loss'))

        trn.run()

    def predict_proba(self, X):
        """

        Args:
            X (wave):
        """
        raise NotImplementedError
