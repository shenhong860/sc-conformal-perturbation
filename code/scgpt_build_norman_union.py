"""Build GEARS-compatible Norman PertData using HVG + perturbation gene union."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from gears import PertData
from scipy.io import mmread

BASE = "/mnt/d/guo/CW1OT/data/norman/"
HVG_FILE = BASE + "norman_hvg1000.h5ad"
ROOT = Path("/mnt/d/guo/CW1OT/results/scgpt_norman")
DATA_DIR = ROOT / "data_union"
MIN_CELLS = 600

S5_TEST_GUIDES = [
    "CEBPE_RUNX1T1__CEBPE_RUNX1T1",
    "TBX3_TBX2__TBX3_TBX2",
    "ETS2_CNN1__ETS2_CNN1",
    "UBASH3B_OSR2__UBASH3B_OSR2",
    "DUSP9_ETS2__DUSP9_ETS2",
    "SET_KLF1__SET_KLF1",
    "SET_CEBPE__SET_CEBPE",
    "LHX1_ELMSAN1__LHX1_ELMSAN1",
    "LYL1_IER5L__LYL1_IER5L",
    "ZC3HAV1_HOXC13__ZC3HAV1_HOXC13",
]


def guide_to_condition(guide: str) -> str:
    side = guide.split("__")[0].split("_")
    real = [t for t in side if not t.startswith("NegCtrl")]
    if not real:
        return "ctrl"
    return "+".join(real)


def main():
    mat = mmread(BASE + "GSE133344_filtered_matrix.mtx.gz").tocsr()
    barcodes = pd.read_csv(
        BASE + "GSE133344_filtered_barcodes.tsv.gz", sep="\t", header=None
    )[0].astype(str).values
    genes = pd.read_csv(BASE + "GSE133344_filtered_genes.tsv.gz", sep="\t", header=None)
    var_names = (
        genes[1].astype(str).values if genes.shape[1] > 1 else genes[0].astype(str).values
    )
    ident = pd.read_csv(BASE + "GSE133344_filtered_cell_identities.csv.gz")
    df = pd.DataFrame({"cell_barcode": barcodes}).merge(
        ident[["cell_barcode", "guide_identity"]], on="cell_barcode", how="left"
    )
    mask = df["guide_identity"].notna().values
    guides = df.loc[mask, "guide_identity"].astype(str).values

    adata_full = ad.AnnData(
        X=mat.T[mask],
        obs=pd.DataFrame({"guide_identity": guides}, index=df.loc[mask, "cell_barcode"].values),
        var=pd.DataFrame(index=var_names),
    )
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)

    hvg = ad.read_h5ad(HVG_FILE)
    hvg_names = list(hvg.var_names.astype(str))
    gene_pos = {g: i for i, g in enumerate(var_names)}

    conditions = np.array([guide_to_condition(g) for g in guides])
    adata_full.obs["condition"] = conditions
    counts = pd.Series(conditions).value_counts()
    keep_conds = counts[counts >= MIN_CELLS].index.tolist()
    keep = adata_full.obs["condition"].isin(keep_conds)
    pert_genes = set()
    for c in keep_conds:
        if c != "ctrl":
            pert_genes.update(c.split("+"))
    missing = sorted(g for g in pert_genes if g not in gene_pos or g not in set(hvg_names))
    missing = [g for g in missing if g in gene_pos]
    union_names = hvg_names + missing
    union_idx = np.array([gene_pos[g] for g in union_names], dtype=int)

    adata = ad.AnnData(
        X=adata_full.X[keep.values][:, union_idx],
        obs=adata_full.obs[keep].copy(),
        var=pd.DataFrame({"gene_name": union_names}, index=union_names),
    )
    adata.obs["cell_type"] = "K562"
    print("union genes", adata.shape, "added", len(missing), missing[:30])
    bad = [c for c in keep_conds if c != "ctrl" and any(g not in set(union_names) for g in c.split("+"))]
    print("conditions with missing genes:", bad)
    keep_conds = [c for c in keep_conds if c not in bad]
    keep2 = adata.obs["condition"].isin(keep_conds)
    adata = adata[keep2].copy()
    print("kept cells", adata.shape, "conditions", len(keep_conds))

    cond_counts = adata.obs.groupby("condition").size().sort_values(ascending=False)
    test = sorted(c for c in (guide_to_condition(g) for g in S5_TEST_GUIDES) if c in cond_counts)
    rest = [c for c in cond_counts.index if c not in test and c != "ctrl"]
    val = rest[:10]
    train = rest[10:] + ["ctrl"]
    print("test", test)
    print("val", val)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pert_data = PertData(str(DATA_DIR))
    pert_data.new_data_process("norman_union", adata=adata)
    pert_data.split = "custom"
    pert_data.set2conditions = {"train": train, "val": val, "test": test}
    (ROOT / "splits_union.json").write_text(
        json.dumps({"train": train, "val": val, "test": test, "genes": union_names}, indent=2),
        encoding="utf-8",
    )
    print("splits saved", ROOT / "splits_union.json")


if __name__ == "__main__":
    main()
