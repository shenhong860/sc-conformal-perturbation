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
    cond_info = {}
    for drug in DRUGS:
        for cell in CELL_TYPES:
            ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
            tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
            if len(ctrl) < 600 or len(tgt) < 600:
                continue
            Xc_tr, Xc_cal, Xc_te = split_arrays(to_np(ctrl.X))
            Xt_tr, Xt_cal, Xt_te = split_arrays(to_np(tgt.X))
            pred_in = Xt_tr.mean(0) - Xc_tr.mean(0)
            sigma_cal = shrink(
                np.sqrt(
                    np.var(Xt_cal, axis=0, ddof=1) / Xt_cal.shape[0]
                    + np.var(Xc_cal, axis=0, ddof=1) / Xc_cal.shape[0]
                )
            )
            true_cal = Xt_cal.mean(0) - Xc_cal.mean(0)
            scores = np.abs(true_cal - pred_in) / (sigma_cal + 1e-6)
            sigma_te = shrink(
                np.sqrt(
                    np.var(Xt_te, axis=0, ddof=1) / Xt_te.shape[0]
                    + np.var(Xc_te, axis=0, ddof=1) / Xc_te.shape[0]
                )
            )
            true_te = Xt_te.mean(0) - Xc_te.mean(0)
            cond_info[(drug, cell)] = {
                "scores": scores,
                "sigma_te": sigma_te,
                "true_te": true_te,
                "pred_in": pred_in,
            }

    cover_rows = []
    for held in DRUGS:
        for cell in CELL_TYPES:
            test_keys = [(d, c) for (d, c) in cond_info if d == held and c == cell]
            cal_keys = [(d, c) for (d, c) in cond_info if d != held and c == cell]
            if not test_keys or not cal_keys:
                continue
            cal_scores = np.concatenate([cond_info[k]["scores"] for k in cal_keys])
            q = quantile_higher(cal_scores)
            pred_out = np.mean([cond_info[k]["pred_in"] for k in cal_keys], axis=0)
            for k in test_keys:
                info = cond_info[k]
                cover = np.abs(info["true_te"] - pred_out) <= q * info["sigma_te"] + 1e-6
                cover_rows.append((held, cell, float(cover.mean())))
                print(held, cell, "coverage", round(float(cover.mean()), 4), flush=True)
    arr = np.array([r[2] for r in cover_rows])
    print("=== SUMMARY ===", flush=True)
    print("overall", round(float(arr.mean()), 4), "n", len(arr), flush=True)
    for cell in CELL_TYPES:
        c = np.array([r[2] for r in cover_rows if r[1] == cell])
        print(cell, round(float(c.mean()), 4), "n", len(c), flush=True)


if __name__ == "__main__":
    main()
