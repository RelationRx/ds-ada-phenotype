import os.path as osp
from typing import Dict

import pandas as pd
import relation_data_lake as rdl
import torch


def get_crispri_il2_screen() -> Dict[str, torch.Tensor]:
    """
    CRISPRi IL2 screen from
    `reference <https://www.biorxiv.org/content/10.1101/2021.05.11.443701v1>`

    Phenotype computed with log-fold change between high/low FACS sorted population

    :return: dictionary of genes to phenotypes
    """
    rdl.pull_folder("parsed/literature_datasets/genedisco_datasets")
    df = pd.read_excel(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "genedisco_datasets",
            "schmidt_crispria_gw_screens.xlsx",
        ),
        sheet_name="CRISPRi.IL2screen.gene_summary",
    )
    df["gene"] = df.id.apply(lambda g: rdl.recognize_gene_id(g))
    df.dropna(inplace=True)
    return dict(
        df.groupby("gene").apply(lambda d: torch.tensor(d["neg|lfc"].median()).float())
    )


def get_crispri_ifng_screen() -> Dict[str, torch.Tensor]:
    """
    CRISPRi IFNg screen from
    `reference <https://www.biorxiv.org/content/10.1101/2021.05.11.443701v1>`

    Phenotype computed with log-fold change between high/low FACS sorted population

    :return: dictionary of genes to phenotypes
    """
    rdl.pull_folder("parsed/literature_datasets/genedisco_datasets")
    df = pd.read_excel(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "genedisco_datasets",
            "schmidt_crispria_gw_screens.xlsx",
        ),
        sheet_name="CRISPRi.IFNGscreen.gene_summary",
    )
    df["gene"] = df.id.apply(lambda g: rdl.recognize_gene_id(g))
    df.dropna(inplace=True)
    return dict(
        df.groupby("gene").apply(lambda d: torch.tensor(d["neg|lfc"].median()).float())
    )


def get_k562_survival_screen() -> Dict[str, torch.Tensor]:
    """
    K562 GW CRISPR survival screen under NK cells environment
    `reference <https://www.frontiersin.org/articles/10.3389/fimmu.2019.02879/full>`_

    :return: dictionary of genes to phenotypes
    """
    rdl.pull_folder("parsed/literature_datasets/genedisco_datasets")
    df = pd.read_csv(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "genedisco_datasets",
            "GSE139313_HighSelection_Gene_Summary.txt",
        ),
        sep="\t",
    )
    df["gene"] = df.Gene.apply(lambda g: rdl.recognize_gene_id(g))
    df.dropna(inplace=True)
    return dict(
        df.groupby("gene").apply(lambda d: torch.tensor(d.beta.median()).float())
    )


def get_t_cell_proliferation_screen() -> Dict[str, torch.Tensor]:
    """
    T-cell proliferation GW CRISPR screen from
    `reference  <https://www.sciencedirect.com/science/article/pii/S0092867418313333>`_

    Phenotype computed through FACS sorting based log-fold change

    :return: dictionary of genes to phenotypes
    """
    rdl.pull_folder("parsed/literature_datasets/genedisco_datasets")
    df = pd.read_excel(
        osp.join(
            rdl.PARSED_LITERATURE_DATASETS_DIR,
            "genedisco_datasets",
            "shifrut_intergrated_gw_screens.xlsx",
        )
    )
    # Change sgRNA mapping to unique gene id
    df["gene"] = df.Gene.apply(lambda g: rdl.recognize_gene_id(g))
    df.dropna(inplace=True)  # removing genes we couldn't map
    return dict(
        df.groupby("gene").apply(lambda d: torch.tensor(d.LFC.median()).float())
    )


if __name__ == "__main__":
    a = get_crispri_il2_screen()
    print("plop")
