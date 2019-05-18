"""
F.sigmoid_cross_entropy に class_weight オプションを付加したもの．
"""
import chainer
from chainer import backend
from chainer import function_node
from chainer.functions.activation import sigmoid
from chainer import utils
from chainer import variable
from chainer.utils import type_check


def _check_class_weight_option(class_weight):
    if class_weight is not None:
        if class_weight.ndim != 1:
            raise ValueError('class_weight.ndim should be 1')
        if class_weight.dtype.kind != 'f':
            raise ValueError('The dtype of class_weight should be \'f\'')
        if isinstance(class_weight, variable.Variable):
            raise ValueError('class_weight should be a numpy.ndarray or '
                             'cupy.ndarray, not a chainer.Variable')

class SigmoidCrossEntropy(function_node.FunctionNode):

    """Sigmoid activation followed by a sigmoid cross entropy loss."""

    ignore_label = -1

    def __init__(self, normalize=True, reduce='mean', class_weight=None):
        self.normalize = normalize
        if reduce not in ('mean', 'no'):
            raise ValueError(
                'only \'mean\' and \'no\' are valid for \'reduce\', but '
                '\'%s\' is given' % reduce)
        self.reduce = reduce
        _check_class_weight_option(class_weight)
        self.class_weight = class_weight
        self.count = None

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 't'))
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i',
            x_type.shape == t_type.shape
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))

        xp = backend.get_array_module(*inputs)
        x, t = inputs
        self.ignore_mask = (t != self.ignore_label)

        # stable computation of the cross entropy.
        '''
            L = - t_0 log [1 / (1 + e^x)] - t_1 log [e^x / (1 + e^x)]
              = (t_0 + t_1) log [1 + e^x] - t_1 x
              = log [1 + e^x] - t_1 x
              = log [1 + e^(-|x|)] + (x>0) * x - t_1 x
              = log [1 + e^(-|x|)] + [(x>0) - t] * x
        '''
        # TODO: 型がやばそう
        weight = 1 if self.class_weight is None else self.class_weight[t]
        loss = -(
            self.ignore_mask * weight *
            (x * (t - (x >= 0)) - xp.log1p(xp.exp(-xp.abs(x)))))

        if not self.reduce == 'mean':
            return utils.force_array(loss.astype(x.dtype)),

        if self.normalize:
            count = xp.maximum(1, self.ignore_mask.sum())
        else:
            count = max(1, len(x))
        self.count = count

        # TODO(takagi): Fix to perform division in a specific dtype. See
        # cupy/cupy#1534.
        return utils.force_array(
            xp.divide(xp.sum(loss), self.count), dtype=x.dtype),

    def backward(self, inputs, grad_outputs):
        x, t = self.get_retained_inputs()
        gy, = grad_outputs
        gx, = SigmoidCrossEntropyGrad(
            self.reduce, self.count, self.ignore_mask, t.data, self.class_weight).apply((x, gy))
        return gx, None


class SigmoidCrossEntropyGrad(function_node.FunctionNode):

    """Sigmoid cross entropy gradient function."""

    def __init__(self, reduce, count, ignore_mask, t, class_weight):
        self.reduce = reduce
        self.count = count
        self.ignore_mask = ignore_mask
        self.t = t
        self.class_weight = class_weight

    def forward(self, inputs):
        self.retain_inputs((0, 1))

        xp = backend.get_array_module(*inputs)
        x, gy = inputs

        y, = sigmoid.Sigmoid().forward((x,))
        # TODO: 型がやばそう
        weight = 1 if self.class_weight is None else self.class_weight[self.t]
        if self.reduce == 'mean':
            # TODO(takagi): Fix to perform division in a specific dtype. See
            # cupy/cupy#1534.
            gx = xp.divide(
                gy * self.ignore_mask * weight * (y - self.t), self.count).astype(
                    y.dtype)
        else:
            gx = (gy * self.ignore_mask * weight * (y - self.t)).astype(y.dtype)

        return gx,

    def backward(self, indexes, grad_outputs):

        # TODO: reflect self.class_weight（計算に自信がないので未実装です・・・）
        assert self.class_weight is None, "Not implemented for not-None class_weight"

        ggx, = grad_outputs
        x, gy = self.get_retained_inputs()
        y = chainer.functions.sigmoid(x)
        yp = y * (1 - y)
        gx = yp * chainer.functions.broadcast_to(gy, yp.shape)
        ggy = y - self.t.astype(y.dtype)
        gx *= self.ignore_mask * ggx
        ggy *= self.ignore_mask * ggx

        if self.reduce == 'mean':
            gx /= self.count
            ggy = chainer.functions.sum(ggy) / self.count

        return gx, ggy


def sigmoid_cross_entropy(x, t, normalize=True, reduce='mean', class_weight=None):
    """Computes cross entropy loss for pre-sigmoid activations.
    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            A variable object holding a matrix whose
            (i, j)-th element indicates the unnormalized log probability of
            the j-th unit at the i-th example.
        t (:class:`~chainer.Variable` or :ref:`ndarray`):
            A variable object holding a matrix whose
            (i, j)-th element indicates a signed integer vector of
            ground truth labels 0 or 1.
            If ``t[i, j] == -1``, corresponding ``x[i, j]`` is ignored.
            Loss is zero if all ground truth labels are ``-1``.
        normalize (bool): Variable holding a boolean value which
            determines the normalization constant. If true, this function
            normalizes the cross entropy loss across all instances. If else,
            it only normalizes along a batch size.
        reduce (str): Variable holding a ``str`` which
            determines whether to reduce the shape of the input.
            If it is ``'mean'``, it computes the sum of cross entropy
            and normalize it according to ``normalize`` option.
            If is is ``'no'``, this function computes cross entropy for each
            instance and does not normalize it (``normalize`` option is
            ignored). In this case, the loss value of the ignored instance,
            which has ``-1`` as its target value, is set to ``0``.
        class_weight (:ref:`ndarray`):
            An array that contains constant weights that will be multiplied
            with the loss values along with the second dimension. The shape of
            this array should be ``(2,)``.
    Returns:
        Variable: A variable object holding an array of the cross entropy.
        If ``reduce`` is ``'mean'``, it is a scalar array.
        If ``reduce`` is ``'no'``, the shape is same as those of ``x`` and
        ``t``.
    .. note::
       This function is differentiable only by ``x``.
    .. admonition:: Example
        >>> x = np.array([[-2.0, 3.0, 0.5], [5.0, 2.0, -0.5]]).\
astype(np.float32)
        >>> x
        array([[-2. ,  3. ,  0.5],
               [ 5. ,  2. , -0.5]], dtype=float32)
        >>> t = np.array([[0, 1, 0], [1, 1, -1]]).astype(np.int32)
        >>> t
        array([[ 0,  1,  0],
               [ 1,  1, -1]], dtype=int32)
        >>> F.sigmoid_cross_entropy(x, t)
        variable(0.25664714)
        >>> F.sigmoid_cross_entropy(x, t, normalize=False)
        variable(0.64161783)
        >>> y = F.sigmoid_cross_entropy(x, t, reduce='no')
        >>> y.shape
        (2, 3)
        >>> y.array
        array([[ 0.126928  ,  0.04858735,  0.974077  ],
               [ 0.00671535,  0.126928  , -0.        ]], dtype=float32)
    """
    return SigmoidCrossEntropy(normalize, reduce, class_weight=class_weight).apply((x, t))[0]
