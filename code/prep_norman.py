import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.io import mmread

base = "/mnt/d/guo/CW1OT/data/norman/"
mat = mmread(base + "GSE133344_filtered_matrix.mtx.gz").tocsr()
barcodes = pd.read_csv(base + "GSE133344_filtered_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).values
genes = pd.read_csv(base + "GSE133344_filtered_genes.tsv.gz", sep="\t", header=None)
print("genes head", genes.head(3).to_string(), flush=True)
ident = pd.read_csv(base + "GSE133344_filtered_cell_identities.csv.gz")
df = pd.DataFrame({"cell_barcode": barcodes}).merge(
    ident[["cell_barcode", "guide_identity"]], on="cell_barcode", how="left"
)
mask = df["guide_identity"].notna().values
var_names = genes[1].astype(str).values if genes.shape[1] > 1 else genes[0].astype(str).values
adata = ad.AnnData(
    X=mat.T[mask],
    obs=pd.DataFrame({"guide_identity": df.loc[mask, "guide_identity"].values}, index=df.loc[mask, "cell_barcode"].values),
    var=pd.DataFrame(index=var_names),
)
print("raw", adata.shape, flush=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat")
adata = adata[:, adata.var.highly_variable].copy()
print("hvg", adata.shape, flush=True)
print("controls", int((adata.obs["guide_identity"].str.contains("NegCtrl")).sum()), flush=True)
print(adata.obs["guide_identity"].value_counts().head(10).to_string(), flush=True)
adata.write("/mnt/d/guo/CW1OT/data/norman/norman_hvg1000.h5ad")
print("saved", flush=True)
