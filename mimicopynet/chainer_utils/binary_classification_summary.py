from chainer import functions as F
from chainer import Variable

def binary_classification_summary(y, t):
    """
    F.classification_summary の binary 版
    （内部で F.classification_summary を使用するかも．しないかも．）
    ----
    Args:
        y (Variable): pre-sigmoid value
        t (Variable or xp.ndarray): true label (0/1), has same shape as y
    Returns:
        P, R, F, support
    """
    assert(y.shape == t.shape)
    """
    xp = y.xp
    zeros = Variable(xp.zeros(y.shape, dtype=y.dtype))
    y_presoftmax = F.stack([zeros, y], axis=-1)
    print(y_presoftmax.shape, t.shape)
    p, r, f, sup = F.classification_summary(y_presoftmax, t)
    return p[1], r[1], f[1], sup
    """
    if isinstance(t, Variable):
        t = t.data
    assert(t.max() <= 1)

    xp = y.xp
    y = (y.data >= 0).astype(xp.int32)

    tp = xp.sum(y * t)
    fp = xp.sum(y * (1 - t))
    fn = xp.sum((1 - y) * t)

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = 2 * p * r / (p + r)

    return p, r, f, xp.bincount(t.ravel())
