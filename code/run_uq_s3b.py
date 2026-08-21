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
SHRINK = 0.5
CALIB_FRACS = [0.25, 0.10, 0.05]


def to_np(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def split_arrays(arr, cal_frac, seed=0):
    rng = np.random.RandomState(seed)
    n = len(arr)
    idx = rng.permutation(n)
    a = int(0.5 * n)
    b = a + int(cal_frac * n)
    return arr[idx[:a]], arr[idx[a:b]], arr[idx[b:]]


def shrink(sigma, lam=SHRINK):
    return lam * sigma + (1 - lam) * np.median(sigma)


def quantile_higher(scores, alpha=ALPHA):
    k = np.ceil((1 - alpha) * (1 + len(scores)))
    return np.sort(scores)[int(min(len(scores) - 1, k))]


def main():
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")
    for cal_frac in CALIB_FRACS:
        per_cond = []
        for drug in DRUGS:
            for cell in CELL_TYPES:
                ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
                tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
                if len(ctrl) < 600 or len(tgt) < 600:
                    continue
                Xc_tr, Xc_cal, Xc_te = split_arrays(to_np(ctrl.X), cal_frac)
                Xt_tr, Xt_cal, Xt_te = split_arrays(to_np(tgt.X), cal_frac)
                pred = Xt_tr.mean(0) - Xc_tr.mean(0)
                sigma_cal = shrink(
                    np.sqrt(
                        np.var(Xt_cal, axis=0, ddof=1) / max(1, Xt_cal.shape[0])
                        + np.var(Xc_cal, axis=0, ddof=1) / max(1, Xc_cal.shape[0])
                    )
                )
                scores = np.abs((Xt_cal.mean(0) - Xc_cal.mean(0)) - pred) / (sigma_cal + 1e-6)
                sigma_te = shrink(
                    np.sqrt(
                        np.var(Xt_te, axis=0, ddof=1) / Xt_te.shape[0]
                        + np.var(Xc_te, axis=0, ddof=1) / Xc_te.shape[0]
                    )
                )
                true_te = Xt_te.mean(0) - Xc_te.mean(0)
                per_cond.append({"cell": cell, "scores": scores, "sigma_te": sigma_te, "err_te": np.abs(true_te - pred)})
        covers = []
        for r in per_cond:
            group_scores = np.concatenate([x["scores"] for x in per_cond if x["cell"] == r["cell"]])
            q = quantile_higher(group_scores)
            covers.append(float((r["err_te"] <= q * r["sigma_te"] + 1e-6).mean()))
        covers = np.array(covers)
        print("cal_frac", cal_frac, "overall", round(float(covers.mean()), 4), flush=True)
        for cell in CELL_TYPES:
            c = covers[[i for i, r in enumerate(per_cond) if r["cell"] == cell]]
            print(" ", cell, round(float(c.mean()), 4), flush=True)


if __name__ == "__main__":
    main()
