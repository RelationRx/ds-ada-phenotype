import os.path as osp

import numpy as np
import pandas as pd
import relation_data_lake as rdl


def get_lincs():
    """
    Gets lincs data which details the log-fold change of ~1000 transcripts
    relative to WT for ~5000 different KOs in 9 cell lines.

    :return: Tuple of lincs_data, lincs_genes and cell_lines. lincs_data
    is a (num_KOs, num_transcripts, num_cell_lines) shape array.
    """
    cell_lines = [
        "A375",
        "A549",
        "AGS",
        "BICR6",
        "ES2",
        "HT29",
        "PC3",
        "U251MG",
        "YAPC",
    ]
    rdl.pull_folder("parsed/literature_datasets/L1000")
    meta = pd.read_parquet(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "L1000",
            "Signature_metadata.parquet",
        )
    )
    xpr = pd.read_parquet(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "L1000",
            "level5_crispr_data.parquet",
        )
    )
    gene = pd.read_parquet(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "L1000",
            "Gene_metadata.parquet",
        )
    )

    genes = list(map(str, gene[gene.feature_space == "landmark"].gene_id.tolist()))
    xpr = xpr.loc[genes].T

    meta = meta[meta.pert_type == "trt_xpr"]
    meta = meta[
        ["sig_id", "cell_iname", "cmap_name", "is_exemplar_sig", "is_hiq", "qc_pass"]
    ]
    meta = meta.dropna()
    meta["cmap_name"] = meta.cmap_name.apply(lambda g: rdl.recognize_gene_id(g))
    meta = meta.dropna()

    # All data for the same gene/cell-line combo (after gene remapping) is collapsed
    # to a single value by taking the mean across replicates
    data = {}
    for (cell, gene), df in meta.groupby(["cell_iname", "cmap_name"]):
        if cell not in data:
            data[cell] = {}
        data[cell][gene] = xpr.loc[df.sig_id.values].values.mean(0)

    per_cell_line_KOs = {k: set(data[k].keys()) for k in cell_lines}
    lincs_genes = list(set.intersection(*list(per_cell_line_KOs.values())))
    meta = meta[meta.cell_iname.isin(cell_lines)]
    meta = meta[meta.cmap_name.isin(lincs_genes)]
    meta = meta[["sig_id", "cell_iname", "cmap_name"]]

    lincs = []
    for k, v in data.items():
        if k in cell_lines:
            lincs.append(np.vstack([v[g] for g in lincs_genes]))
    lincs = np.stack(lincs, 0)
    lincs = lincs.transpose(1, 2, 0)
    return lincs, lincs_genes, cell_lines


def get_achilles_screen(scale_to_landmark_genes=True):
    """
    Gets Achilles screen cell line, knockout phenotypic screen
    :param scale_to_landmark_genes: boolean describing whether to perform an
    analogous scaling to the one described in Dempster et al (2018)
    (https://www.biorxiv.org/content/10.1101/720243v1.full.pdf) using essential
    and non essential human survival genes defined from Hart et al. (2015) and
    Blomen et al. (2014). If True then data is scaled and shifter per cell line
    such that essential genes have a median log-fold change of -1 and non-essential
    genes have a median of zero.
    :return: dataframe with columns 'gene', 'context', 'phenotype' annd 'value'
    """
    df = (
        rdl.phenotype_screen_get_data(
            dataset_name="achilles",
            include_control_measurements=False,
            include_control_perturbations=True,
            include_measurements=["lfc"],
        )
        .compute()
        .astype("float32")
    )
    (
        perturbation_metadata,
        measurement_metadata,
        landmarks,
    ) = rdl.phenotype_screen_get_metadata(
        dataset_name="achilles",
        include_control_measurements=False,
        include_control_perturbations=True,
        include_measurements=["lfc"],
        get_landmark_genes=scale_to_landmark_genes,
    )
    assert all(df.index == perturbation_metadata.index)
    df["gene"] = perturbation_metadata["target"]
    df = df.groupby("gene").median(0)
    # There is only one cell lines that appears multiple times: CH-001172,
    # it's 2 columns have a strong pearson of 0.85106226
    df = df.groupby(df.columns.str.split("_").str[0], axis=1).median()

    # scale non essential genes to have a median of zero and essential genes to have a median of -1
    if scale_to_landmark_genes:
        for key, value in landmarks.items():
            landmarks[key] = value[value["gene"].isin(df.index.values)]
        df = (
            -(
                df
                - df.loc[landmarks["nonessentials"]["gene"].values]
                .median(axis=0)
                .values[None, :]
            )
            / df.loc[landmarks["essentials"]["gene"].values]
            .median(axis=0)
            .values[None, :]
        )
    return df


def get_sirna_screen():
    """
    siRNA screen to measure the coordination of cell size and RNA production
    `reference  <https://www.nature.com/articles/s41597-021-00944-5>`_

    We use the phenotype "nuclei_mean_EU_median",
    which corresponds to the median EU signal (measuring nascent transcription)
    per cell in a well

    :return: dataframe with columns 'gene', 'phenotype' annd 'value'
    """
    rdl.pull_folder("parsed/literature_datasets/Muller_HCI-siRNA")
    df = pd.read_parquet(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "Muller_HCI-siRNA",
            "Muller_HCI-siRNA_data.parquet",
        )
    )
    df_metadata = pd.read_parquet(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "Muller_HCI-siRNA",
            "Muller_HCI-siRNA_well_level_metadata.parquet",
        )
    )
    df = df[["nuclei_mean_EU_median"]]
    df["gene_name"] = df_metadata["target_gene_names"]
    df = df.mask(df["gene_name"].eq("None")).dropna()
    # Change sgRNA mapping to unique gene id
    df["mapped_gene_name"] = df.gene_name.apply(lambda g: rdl.recognize_gene_id(g))
    df.dropna(inplace=True)  # removing genes we couldn't map
    df = (
        df.groupby("mapped_gene_name")
        .apply(lambda d: d["nuclei_mean_EU_median"].median())
        .to_frame("value")
        .astype("float32")
        .reset_index()
        .rename({"mapped_gene_name": "gene"}, axis=1)
    )
    df["phenotype"] = "sirna"
    return df
