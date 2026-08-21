import numpy as np
import anndata as ad

DRUGS = [
    "belinostat",
    "dacinostat",
    "givinostat",
    "hesperadin",
    "tanespimycin",
    "jnj_26854165",
    "tak_901",
    "flavopiridol_hcl",
    "alvespimycin_hcl",
]
CELL_TYPES = ["A549", "MCF7", "K562"]
ALPHA = 0.05


def to_np(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def split_arrays(arr, fracs=(0.5, 0.25, 0.25), seed=0):
    rng = np.random.RandomState(seed)
    n = len(arr)
    idx = rng.permutation(n)
    a = int(fracs[0] * n)
    b = a + int(fracs[1] * n)
    return arr[idx[:a]], arr[idx[a:b]], arr[idx[b:]]


def main():
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")
    rows = []
    for drug in DRUGS:
        for cell in CELL_TYPES:
            ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
            tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
            if len(ctrl) < 600 or len(tgt) < 600:
                continue
            Xc_tr, Xc_cal, Xc_te = split_arrays(to_np(ctrl.X))
            Xt_tr, Xt_cal, Xt_te = split_arrays(to_np(tgt.X))
            pred = Xt_tr.mean(0) - Xc_tr.mean(0)
            sigma_cal = np.sqrt(
                np.var(Xt_cal, axis=0, ddof=1) / Xt_cal.shape[0]
                + np.var(Xc_cal, axis=0, ddof=1) / Xc_cal.shape[0]
            )
            true_cal = Xt_cal.mean(0) - Xc_cal.mean(0)
            scores = np.abs(true_cal - pred) / (sigma_cal + 1e-6)
            q = np.quantile(scores, min(1.0, np.ceil((1 - ALPHA) * (1 + len(scores))) / len(scores)))
            sigma_te = np.sqrt(
                np.var(Xt_te, axis=0, ddof=1) / Xt_te.shape[0]
                + np.var(Xc_te, axis=0, ddof=1) / Xc_te.shape[0]
            )
            true_te = Xt_te.mean(0) - Xc_te.mean(0)
            cover = np.abs(true_te - pred) <= q * sigma_te + 1e-6
            width = q * sigma_te
            rows.append(
                {
                    "drug": drug,
                    "cell": cell,
                    "n_genes": len(cover),
                    "coverage": float(cover.mean()),
                    "width_med": float(np.median(width)),
                }
            )
            print(
                drug,
                cell,
                "coverage",
                round(float(cover.mean()), 4),
                "width_med",
                round(float(np.median(width)), 5),
                flush=True,
            )
    print("=== SUMMARY ===", flush=True)
    cov = np.array([r["coverage"] for r in rows])
    print("overall", round(float(cov.mean()), 4), "n_conditions", len(rows), flush=True)
    for cell in CELL_TYPES:
        c = np.array([r["coverage"] for r in rows if r["cell"] == cell])
        print(cell, round(float(c.mean()), 4), "n", len(c), flush=True)


if __name__ == "__main__":
    main()
