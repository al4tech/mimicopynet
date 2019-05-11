from chainer.functions.evaluation import classification_summary
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter

from chainer.links import Classifier


class PRFClassifier(Classifier):
    compute_accuracy = True
    # OVERRIDE
    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=classification_summary.classification_summary,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(Classifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor
    # OVERRIDE
    def forward(self, *args, **kwargs):
        """
        P，R，F を report する Classifier.
        二値分類を想定しています．

        Classifier の init 時に accfun を指定しますが，
        これは通常だと accuracy (Varaible) を返すようにしますが，
        この Classifier を使う場合は，
        (P, R, F, support) （いずれも Variable）を返すようにしてください
        （参考：F.classification_summary）

        ----
        （以下は元々の Classifier に関する説明）
    
        Computes the loss value for an input and label pair.
        It also computes accuracy and stores it to the attribute.
        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.
        When ``label_key`` is ``int``, the correpoding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground truth
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.
        .. note::
            We set ``None`` to the attributes ``y``, ``loss`` and ``accuracy``
            each time before running the predictor, to avoid unnecessary memory
            consumption. Note that the variables set on those attributes hold
            the whole computation graph when they are computed. The graph
            stores interim values on memory required for back-propagation.
            We need to clear the attributes to free those values.
        Returns:
            ~chainer.Variable: Loss value.
        """

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            # self.accuracy = self.accfun(self.y, t)
            # reporter.report({'accuracy': self.accuracy}, self)
            p, r, f, support = self.accfun(self.y, t)
            reporter.report({'precision': p}, self)
            reporter.report({'recall': r}, self)
            reporter.report({'fvalue': f}, self)
        return self.loss
