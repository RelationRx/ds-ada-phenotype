import numpy as np


def truncate_k(x, k, geq=True):
    m, _ = x.shape
    rows, _ = np.indices((m, k))
    # get (unsorted) indices of top-k values
    if geq:
        k_indices = np.argpartition(x, -k, axis=1)[:, -k:]
        kth_vals = x[rows, k_indices].min(axis=1)
        is_kth = x >= kth_vals[:, None]
    else:
        k_indices = np.argpartition(x, k, axis=1)[:, :k]
        kth_vals = x[rows, k_indices].max(axis=1)
        is_kth = x <= kth_vals[:, None]

    return is_kth
