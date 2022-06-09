from typing import Callable

import numpy as np
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy.stats import fisher_exact as fish
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def primary_dists(X: np.ndarray, metric: str = "euclidean"):
    """
    For some (N,M,C) array with N data points, M features and C experiments,
    creats a (C,N,N) array where each row (0th axis) details pairwise distances
    between the datapoints in that experiment according to some input metric.

    :param X: input (N,M,C) array
    :param metric: metric with which to perform distance calculations

    :return: (C,N,N) shape array detailing pairwise distances across
    experiments
    """
    dists = {}
    for idx, cell_line in enumerate(np.arange(X.shape[-1])):
        dists[cell_line] = cdist(X[:, :, idx], X[:, :, idx], metric)
    return np.stack([dists[cell_line] for cell_line in np.arange(X.shape[-1])])


def simcos_single_cell_line(
    X: np.ndarray,
    s: int,
    primary_metric: str = "euclidean",
    tfm: Callable = lambda x: np.arccos(x),
):
    """
    Perform shared nearest neighbour distance calculations for some input (M,C)
    shape array using some primary distance metric before performing inverse
    cos of the output to create a distance metric which respects the triangle
    inequality.
    :param X: input (M,C) array
    :param s: size of nearest neighbour graphs to consider
    :param primary_metric: metric with which to perform distance calculations
    :param tfm: output transform to apply to calculated shared nearest
    neighbour distances. Defaults to arcos

    :return: (N,N) shape array detailing pairwise shared nearest neighbour
    distances across experiments
    """
    nbrs = NearestNeighbors(
        n_neighbors=s, algorithm="ball_tree", metric=primary_metric
    ).fit(X)
    _, indices = nbrs.kneighbors(X)
    holder = np.full((X.shape[0], X.shape[0]), False)
    for idx, row in enumerate(indices):
        holder[idx, row] = True

    dists = holder[:, None, :] & holder[None, :, :]

    return tfm(np.sum(dists, axis=-1) / s)


def simcos(
    X: np.ndarray,
    s: int,
    primary_metric: str = "euclidean",
    tfm: Callable = lambda x: np.arccos(x),
):
    """
    Perform shared nearest neighbour distance calculations for some
    input (N,M,C) shape array using some primary distance metric before
    performing inverse cos of the output to create a distance metric
    which respects the triangle inequality.
    :param X: input (N,M,C) array
    :param s: size of nearest neighbour graphs to consider
    :param primary_metric: metric with which to perform distance
    calculations
    :param tfm: output transform to apply to calculated shared nearest
    neighbour distances. Defaults to arcos

    :return: (C,N,N) shape array detailing pairwise shared nearest
    neighbour distances across experiments
    """
    similarities = {}
    for cidx, cell_line in tqdm(enumerate(np.arange(X.shape[-1]))):
        similarities[cell_line] = simcos_single_cell_line(
            X[:, :, cidx], s, primary_metric, tfm
        )
    return np.stack([similarities[cell_line] for cell_line in np.arange(X.shape[-1])])


def vectorized_make_crosstab(arr: np.ndarray, n: int = 2):
    """
    Given some (M,N) binary ({0,1}) or trinary ({-1,0,1}) array arr, forms
    a (K,M,M) array where the zeroth axis is a flattened contingency matrix
    detailing the overlaps of each pair of rows in arr

    :param arr: 2d array
    :param n: Whether to form binary or trinary flattened contingency matrices
    :return: (n^2,M,M) array
    """
    if n == 2:
        n1 = np.sum((arr[:, None, :] != 0) & (arr[None, :, :] != 0), axis=-1)
        n2 = np.sum((arr[:, None, :] != 0) & (arr[None, :, :] == 0), axis=-1)
        n3 = np.sum((arr[:, None, :] == 0) & (arr[None, :, :] != 0), axis=-1)
        n4 = np.sum((arr[:, None, :] == 0) & (arr[None, :, :] == 0), axis=-1)

        return np.stack([n1, n2, n3, n4])
    elif n == 3:
        n1 = np.sum((arr[:, None, :] < 0) & (arr[None, :, :] < 0), axis=-1)
        n2 = np.sum((arr[:, None, :] < 0) & (arr[None, :, :] == 0), axis=-1)
        n3 = np.sum((arr[:, None, :] < 0) & (arr[None, :, :] > 0), axis=-1)
        n4 = np.sum((arr[:, None, :] == 0) & (arr[None, :, :] < 0), axis=-1)
        n5 = np.sum((arr[:, None, :] == 0) & (arr[None, :, :] == 0), axis=-1)
        n6 = np.sum((arr[:, None, :] == 0) & (arr[None, :, :] > 0), axis=-1)
        n7 = np.sum((arr[:, None, :] > 0) & (arr[None, :, :] < 0), axis=-1)
        n8 = np.sum((arr[:, None, :] > 0) & (arr[None, :, :] == 0), axis=-1)
        n9 = np.sum((arr[:, None, :] > 0) & (arr[None, :, :] > 0), axis=-1)

        return np.stack([n1, n2, n3, n4, n5, n6, n7, n8, n9])


def test_ordinal(vec: np.ndarray):
    """
    Perform a linear by linear association test on trinarized contingency table
    and output the p value as well as the enrichment of DEG overlaps over expected.

    :param vec: flattened contingency matrix i.e. a shape (9,) array in which the
    first three values correspond to row 1 of the contingency, second three values
    to the second row and final threevalues to the final row

    :return: List containing
    """
    tab = sm.stats.Table(vec.reshape((3, 3)))
    rslt = tab.test_nominal_association()
    return [
        rslt.pvalue,
        (tab.table_orig[0, 0] + tab.table_orig[2, 2])
        / (tab.fittedvalues[0, 0] + tab.fittedvalues[2, 2]),
    ]


def vectorized_DEG_crosstab_distance(arr: np.ndarray, n: int, eps: float, alt: str):
    """
    In a vectorized manner, perform statistical analysis of overlap of DEGs between
    KOs using contingency tables.

    :param arr: input (M,N) array
    :param n: Integer, either 2 or 3, describing whether to binarize or trinarize the
    input data
    :param eps: float to add to the contingency table if any row or column sums are
    zero - otherwise we're unable to create an expected contingency matrix under the
    null assumption of no ordinal relation between KOs
    :param alt: If performing Fisher's exact test, which alternative hypothesis to
    use. Defaults to greater i.e. p-values correspond to the chance of observing this
    many or more DEG overlaps under a random contingency matrix

    :return: tuple of the statistic (odds-ratio) and the p-value
    """
    ctab = vectorized_make_crosstab(arr, n=n)
    if n == 2:
        out = np.apply_along_axis(
            lambda x: fish(np.reshape(x, (n, n)), alternative=alt), 0, ctab
        )
        stat, p = out[0, :, :], out[1, :, :]
        stat[np.diag_indices(stat.shape[0])] = 1
    else:
        out = np.apply_along_axis(test_ordinal, 0, ctab + eps)

        stat, p = out[0, :, :], out[1, :, :]
        stat[np.diag_indices(stat.shape[0])] = 0

    p[np.diag_indices(p.shape[0])] = 1
    return stat, p


def get_consistency_single_KO(
    knn_slice: np.ndarray, f: float = 0.5, C: int = 9, minpartners: int = 1
):
    """
    For a set of k-NN indices centred at a single KO from multiple cell lines,
    identify the minimal k such that there exists at least {minpartners} other
    KOs which appear in more than some fraction f of cell lines.
    :param knn_slice: input (C*(M-1),) flattened array where each M-1 numbers row
    corresponds to the (ordered)
    nearest neighbour indices of the input KO in a given cell line
    :param f: Fraction of of cell lines needed for a KO to be considered a consistent
    close KO
    :param C: Number of cell lines - used for reshaping the flattened input array
    :param minpartners: Number of partner KOs which is needed for a local
    neighbourhood to be considered 'stable'

    :return: the minimal k
    """
    knn_slice = knn_slice.reshape((C, -1))
    for k in np.arange(knn_slice.shape[-1]):
        _, counts = np.unique(knn_slice[:, :k], return_counts=True)
        if np.sum(counts > f * C) >= minpartners:
            return k


def get_consistency_from_knn_indices(
    knn_indices: np.ndarray, f: float = 0.5, C: int = 9, minpartners: int = 1
):
    """
    For a set of k-NN indices from multiple cell lines, identify the minimal k
    such that there exists at least {minpartners} other KOs which appear in
    more than some fraction f of cell lines.
    :param knn_slice: input (M,C*(M-1)) flattened array where each M numbers
    row corresponds to the (ordered)
    nearest neighbour indices of the input KO in a given cell line
    :param f: Fraction of of cell lines needed for a KO to be considered a
    consistent close KO
    :param C: Number of cell lines - used for reshaping the flattened input
    array
    :param minpartners: Number of partner KOs which is needed for a local
    neighbourhood to be considered 'stable'

    :return: the minimal k for each KO as a fraction of the total number of data
    points
    """
    out = np.apply_along_axis(
        lambda x: get_consistency_single_KO(x, f=f, C=C, minpartners=minpartners),
        1,
        knn_indices,
    )

    return out / knn_indices.shape[0]


def get_consistency(
    arr: np.ndarray, metric: str = "euclidean", f: float = 0.5, minpartners: int = 1
):
    """
    For some input (M,N,C) shape array detailing data for M KOs across C cell
    lines, identify the consistency of local neighbourhoods centred around each KO.
    Specifically, for each KO, identify the minimal k such that k-NN graphs centred
    around that KO across cell lines contain at least {minpartners} other KOs which
    appear in the neighbourhood set in at least some fraction {f} of cell lines.

    :param arr: input (M,N,C) detailing data for each of the cell lines and KOs
    :param metric: which metric to use when creating the k-NN graph - must be an sklearn
    accepted metric
    :param f: Fraction of of cell lines needed for a KO to be considered a consistent
    close KO
    :param C: Number of cell lines - used for reshaping the flattened input array
    :param minpartners: Number of partner KOs which is needed for a local neighbourhood
    to be considered 'stable'

    :return: the minimal k for each KO as a fraction of the total number of data points
    (i.e. k/M)
    """
    # creating a full NN graph for each cell line - may take some time with the full dataset
    nbrs = {
        idx: NearestNeighbors(
            n_neighbors=arr.shape[0], algorithm="ball_tree", metric=metric
        ).fit(arr[:, :, idx])
        for idx in np.arange(arr.shape[-1])
    }
    indices = {}
    for idx in np.arange(arr.shape[-1]):
        _, indices[idx] = nbrs[idx].kneighbors(arr[:, :, idx])

    indices = np.concatenate(
        [indices[idx][:, 1:] for idx in np.arange(arr.shape[-1])], axis=1
    )

    return get_consistency_from_knn_indices(
        indices, f=f, C=arr.shape[-1], minpartners=minpartners
    )
