from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from ..tools.data_loading import get_lincs
from ..tools.metrics import vectorized_DEG_crosstab_distance

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        default=3,
        help="trinary or binary contingency matrix - defaults to 3",
    )
    parser.add_argument(
        "--eps",
        default=1e-6,
        help="tolerance to add to contingency tables for contingency testing",
    )
    args = parser.parse_args()

    THRESH = 1.96
    lincs, lincs_genes, cell_lines = get_lincs()
    lincs[abs(lincs) < THRESH] = 0
    lincs[lincs > THRESH] = 1
    lincs[lincs < -THRESH] = -1

    lincs_shuffled = np.copy(lincs)
    for cell_line in np.arange(lincs.shape[-1]):
        order = np.arange(lincs.shape[0])
        np.random.shuffle(order)
        lincs_shuffled[:, :, cell_line] = lincs[order, :, cell_line]

    if args.n == 2:
        test = "fisher"
    else:
        test = "linear_by_linear"

    def single_arg_vectorized_DEG_crosstab_distance(arr):
        out = vectorized_DEG_crosstab_distance(arr, args.n, args.eps, "greater")
        return out

    for name, arr in zip(["real", "shuffled"], [lincs, lincs_shuffled]):
        print(f"{name} data:")
        OR_all = []
        p_all = []

        p = Pool()
        p_outs = p.imap(
            single_arg_vectorized_DEG_crosstab_distance,
            tqdm([arr[:, :, idx] for idx in np.arange(arr.shape[-1])]),
        )
        for p_out in p_outs:
            OR, p = p_out
            OR_all.append(OR)
            p_all.append(p)

        np.save(f"LINCS_{name}_{test}_OR.npy", np.stack(OR_all).astype("float"))
        np.save(f"LINCS_{name}_{test}_p.npy", np.stack(p_all).astype("float"))
