import numpy as np


def truncate_top_k(x, k):
    m, _, levs = x.shape
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]

    # get k-th value
    rows, _, levels = np.indices((m, k, levs))
    kth_vals = x[rows, topk_indices, levels].min(axis=1)

    # get boolean mask of greater smaller than k-th
    is_geq_than_kth = x >= kth_vals[:, None, :]

    return is_geq_than_kth
