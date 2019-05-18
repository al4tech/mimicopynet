import numpy as np
import sys
from chainer import Chain, ChainList, cuda, gradient_check, Function, Link, \
    optimizers, serializers, utils, Variable, datasets, using_config, training, iterators
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L

from ...data import MidiDataset
from ...chainer_utils import f_measure_accuracy, PRFClassifier, binary_classification_summary, sigmoid_cross_entropy

class Net(Chain):
    def __init__(self, num_channel=None, wave_samples=None, score_samples=None, class_weight=None):
        self.num_channel = num_channel
        self.wave_samples = wave_samples
        self.score_samples = score_samples
        self.class_weight = class_weight
        assert self.wave_samples % self.score_samples == 0, 'wave_samples should be multiple of score_samples'
        super(Net, self).__init__(
            c1=L.Convolution1D(1, 16, ksize=7, dilate=1, pad=1 * 3),
            # b1=L.BatchNormalization(16),
            c2=L.Convolution1D(16, 16, ksize=7, dilate=4, pad=4 * 3),
            # b2=L.BatchNormalization(16),
            c3=L.Convolution1D(16, 16, ksize=7, dilate=16, pad=16 * 3),
            # b3=L.BatchNormalization(16),
            c4=L.Convolution1D(16, 16, ksize=7, dilate=64, pad=64 * 3),
            # b4=L.BatchNormalization(16),
            c5=L.Convolution1D(16, 16, ksize=7, dilate=256, pad=256 * 3),
            # b5=L.BatchNormalization(16),
            c6=L.Convolution1D(16, 16, ksize=7, dilate=1024, pad=1024 * 3),
            c7=L.Convolution1D(16, 128, ksize=7, dilate=4096, pad=4096 * 3),
            l1=L.Linear(128, 128),
            l2=L.Linear(128, 128)
        )
        self.cnt = 0
    def __call__(self, X):
        """
        Args:
            X (Variable):
                音声波形．shape==(*, num_channel, wave_samples) dtype==xp.float32
                値は ±1 に収まる程度のスケール．
        Returns:
            y (Variable):
                「各音が各時刻に鳴っているかどうか」の判定結果 (pre_sigmoid value)
                 shape==(*, 128, score_samples) dtype==xp.float32
        """
        assert X.shape[1:] == (self.num_channel, self.wave_samples)
        h = X
        h = F.relu(self.c1(h)) + h  # (bs, 16, 44100) + (bs, 1, 44100) is possible
        # h = self.b1(h)
        h = F.relu(self.c2(h)) + h
        # h = self.b2(h)
        h = F.relu(self.c3(h)) + h
        # h = self.b3(h)
        h = F.relu(self.c4(h)) + h
        # h = self.b4(h)
        h = F.relu(self.c5(h)) + h
        # h = self.b5(h)
        h = F.relu(self.c6(h)) + h
        h = F.relu(self.c7(h))

        # Pooling
        stride = self.wave_samples // self.score_samples  # 割り切れることを assert してある．
        h = F.max_pooling_1d(h, ksize=stride, pad=0, stride=stride)

        assert h.shape[1:] == (128, self.score_samples)

        # Channelwise FC
        h = F.swapaxes(h, 1, 2)
        h = F.reshape(h, [-1, 128])
        h = F.relu(self.l1(h))
        h = self.l2(h)
        h = F.reshape(h, [-1, self.score_samples, 128])

        y = F.swapaxes(h, 1, 2)
        assert y.shape[1:] == (128, self.score_samples)
        self.cnt += 1
        return y

    # @staticmethod
    def lossfun(self, y, t):
        """
        Args:
            y (Variable) 上参照
            t (Variable) shape==(*, 2, 128, score_samples)) dtype==xp.int32
        """
        return sigmoid_cross_entropy(y, t[:, 0, :, :], class_weight=self.class_weight)

    # @staticmethod
    def accfun(self, y, t):
        """
        Args:
            y (Variable) 上参照
            t (Variable) shape==(*, 2, 128, score_samples)) dtype==xp.int32
        Returns:
            (P, R, F, support)
                それぞれ type は xp.ndarray
                P, R, F : shape==(), dtype==xp.float64
                support : shape==(2,), dtype==xp.int64
        """
        return binary_classification_summary(y, t[:, 0, :, :])


class NoCQTModel(object):
    def __init__(self):
        pass

    def fit(self, midis, eval_midis=None, **kwargs):
        """
        Args:
            midis (MidiDataset):

            eval_midis (None / MidiDataset):

        """

        # デフォルトの設定．
        conf = {
            'bs_train': 10,
            'bs_eval': 10,
            'gpu': -1,
            'result_dir': 'result',
            'num_epoch': 100,
            'num_channel': 1,
            'class_weight': None
        }
        # train_size = 10000
        # eval_size = 1000
        # length_sec = 0.5
        # no_drum = True
        # num_part = 1

        # 設定の上書き．
        for k, v in kwargs.items():
            conf[k] = v

        # データセットの準備．
        if isinstance(midis, MidiDataset):
            dataset_train = midis
        else:
            raise ValueError
            # dataset_train = MidiDataset(midis, train_size, length_sec=length_sec, no_drum=no_drum, num_part=num_part)
        if eval_midis is not None:
            if isinstance(eval_midis, MidiDataset):
                dataset_eval = eval_midis
            else:
                raise ValueError
                # dataset_eval = MidiDataset(eval_midis, eval_size, length_sec=length_sec, no_drum=no_drum, num_part=num_part)

        # データセットが繰り出してくるデータの shape を確認
        wave_shape, score_feats_shape = dataset_train.get_shape()

        # Chain の準備
        net = Net(num_channel=wave_shape[0], wave_samples=wave_shape[1], score_samples=score_feats_shape[2], class_weight=conf['class_weight'])
        mdl = PRFClassifier(net, lossfun=net.lossfun, accfun=net.accfun)
        if conf['gpu'] >= 0:
            cuda.get_device(conf['gpu']).use()
            mdl.to_gpu()
        opt = optimizers.MomentumSGD(lr=0.001).setup(mdl)
        itr_train = iterators.SerialIterator(dataset_train, shuffle=False, batch_size=conf['bs_train'])
        upd = training.StandardUpdater(itr_train, opt, device=conf['gpu'])
        trn = training.Trainer(upd, (conf['num_epoch'], 'epoch'), out=conf['result_dir'])
        trn.extend(extensions.LogReport(trigger=(10, 'iteration')))
        trn.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
        trn.extend(extensions.snapshot_object(mdl, filename='model_epoch-{.updater.epoch}'))
        if eval_midis is not None:
            itr_eval = iterators.SerialIterator(dataset_eval, shuffle=False, repeat=False, batch_size=conf['bs_eval'])
            trn.extend(extensions.Evaluator(itr_eval, mdl, device=conf['gpu']))
        # trn.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
        trn.extend(extensions.PrintReport(['epoch', 'iteration', 'main/loss', 'main/precision', 'main/recall', 'main/fvalue', 'validation/main/loss', 'validation/main/precision', 'validation/main/recall', 'validation/main/fvalue', 'elapsed_time']))
        trn.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
        trn.extend(extensions.PlotReport(['main/fvalue', 'validation/main/fvalue'], x_key='epoch', file_name='accuracy.png'))
        trn.extend(extensions.dump_graph('main/loss'))

        trn.run()

    def predict_proba(self, X):
        """

        Args:
            X (wave):
        """
        raise NotImplementedError
