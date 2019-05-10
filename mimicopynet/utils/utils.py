import numpy as np

def get_array_in_fixed_size(src, axis, size_dest, start_src):
    """
    src の axis 方向から size_dest の長さだけ取り出した array を返します．
    端はゼロパディングされ，必ず（その方向に）size_dest の長さの配列が返ります．
    --------
    Args:
        src (numpy.ndarray or .h5 handle):
            ソースの配列．
        axis (int):
            切り出す軸のインデックス．
        size_dest (int):
            切り出すサイズ．0以上の値である必要がある．
        start_src (int):
            切り出す始点．負の値でも良いし，src.shape[axis] 以上の値でも良い．
    Returns:
        dest (numpy.ndarray):
            配列を切り出した結果．
            axis 方向のサイズは size_dest となる．
            それ以外の方向のサイズは，ソース配列と同じになる．
            dtype はソース配列と同じになる．
    """
    shape_src, dtype_src = src.shape, src.dtype

    size_src = shape_src[axis]
    dest = np.zeros([size_dest if i==axis else n for i, n in enumerate(shape_src)], dtype=dtype_src)
    # if start_src is None:
    #     start_src = np.random.randint(size_src + size_dest - 1) - size_dest + 1
    end_src = start_src + size_dest
    start_dest = 0
    end_dest = size_dest

    if start_src >= size_src:
        return dest
    elif start_src < 0:
        start_dest -= start_src
        start_src = 0

    if end_src <= 0:
        return dest
    elif end_src > size_src:
        end_dest -= (end_src - size_src)
        end_src = size_src

    idx_src = tuple(
        slice(start_src, end_src) if i==axis else slice(None)
        for i in range(len(shape_src))
    )
    idx_dest = tuple(
        slice(start_dest, end_dest) if i==axis else slice(None)
        for i in range(len(shape_src))
    )
    dest.__setitem__(idx_dest, src.__getitem__(idx_src))
    return dest