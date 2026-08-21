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


def to_np(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def split_arrays(arr, fracs=(0.5, 0.25, 0.25), seed=0):
    rng = np.random.RandomState(seed)
    n = len(arr)
    idx = rng.permutation(n)
    a = int(fracs[0] * n)
    b = a + int(fracs[1] * n)
    return arr[idx[:a]], arr[idx[a:b]], arr[idx[b:]]


def shrink(sigma, lam=SHRINK):
    return lam * sigma + (1 - lam) * np.median(sigma)


def quantile_higher(scores, alpha=ALPHA):
    k = np.ceil((1 - alpha) * (1 + len(scores)))
    return np.sort(scores)[int(min(len(scores) - 1, k))]


def main():
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")
    seeds = list(range(5))
    overall_by_seed = []
    cell_by_seed = {cell: [] for cell in CELL_TYPES}
    for seed in seeds:
        per_cond = []
        for drug in DRUGS:
            for cell in CELL_TYPES:
                ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
                tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
                if len(ctrl) < 600 or len(tgt) < 600:
                    continue
                Xc_tr, Xc_cal, Xc_te = split_arrays(to_np(ctrl.X), seed=seed)
                Xt_tr, Xt_cal, Xt_te = split_arrays(to_np(tgt.X), seed=seed)
                pred = Xt_tr.mean(0) - Xc_tr.mean(0)
                sigma_cal = shrink(
                    np.sqrt(
                        np.var(Xt_cal, axis=0, ddof=1) / Xt_cal.shape[0]
                        + np.var(Xc_cal, axis=0, ddof=1) / Xc_cal.shape[0]
                    )
                )
                true_cal = Xt_cal.mean(0) - Xc_cal.mean(0)
                scores = np.abs(true_cal - pred) / (sigma_cal + 1e-6)
                sigma_te = shrink(
                    np.sqrt(
                        np.var(Xt_te, axis=0, ddof=1) / Xt_te.shape[0]
                        + np.var(Xc_te, axis=0, ddof=1) / Xc_te.shape[0]
                    )
                )
                true_te = Xt_te.mean(0) - Xc_te.mean(0)
                per_cond.append(
                    {
                        "cell": cell,
                        "scores": scores,
                        "sigma_te": sigma_te,
                        "err_te": np.abs(true_te - pred),
                    }
                )
        covers = []
        for r in per_cond:
            group_scores = np.concatenate([x["scores"] for x in per_cond if x["cell"] == r["cell"]])
            q = quantile_higher(group_scores)
            cover = r["err_te"] <= q * r["sigma_te"] + 1e-6
            covers.append(float(cover.mean()))
        covers = np.array(covers)
        print("seed", seed, "overall", round(float(covers.mean()), 4), flush=True)
        overall_by_seed.append(float(covers.mean()))
        for cell in CELL_TYPES:
            c = covers[[i for i, r in enumerate(per_cond) if r["cell"] == cell]]
            cell_by_seed[cell].append(float(c.mean()))
    o = np.array(overall_by_seed)
    print("MEAN_OVERALL", round(float(o.mean()), 4), "CI95", round(float(1.96 * o.std(ddof=1) / np.sqrt(len(o))), 4), flush=True)
    for cell in CELL_TYPES:
        c = np.array(cell_by_seed[cell])
        print("CELL", cell, round(float(c.mean()), 4), "CI95", round(float(1.96 * c.std(ddof=1) / np.sqrt(len(c))), 4), flush=True)


if __name__ == "__main__":
    main()
