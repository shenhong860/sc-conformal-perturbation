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
TOPK = 50


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


def precision_at_k(pred_rank, true_rank, k=TOPK):
    pred_set = set(pred_rank[:k])
    true_set = set(true_rank[:k])
    return len(pred_set & true_set) / k


def main():
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")
    prec_raw, prec_filt = [], []
    for drug in DRUGS:
        for cell in CELL_TYPES:
            ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
            tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
            if len(ctrl) < 600 or len(tgt) < 600:
                continue
            Xc_tr, Xc_cal, Xc_te = split_arrays(to_np(ctrl.X))
            Xt_tr, Xt_cal, Xt_te = split_arrays(to_np(tgt.X))
            pred = Xt_tr.mean(0) - Xc_tr.mean(0)
            sigma_cal = shrink(
                np.sqrt(
                    np.var(Xt_cal, axis=0, ddof=1) / Xt_cal.shape[0]
                    + np.var(Xc_cal, axis=0, ddof=1) / Xc_cal.shape[0]
                )
            )
            scores = np.abs((Xt_cal.mean(0) - Xc_cal.mean(0)) - pred) / (sigma_cal + 1e-6)
            q = quantile_higher(scores)
            sigma_te = shrink(
                np.sqrt(
                    np.var(Xt_te, axis=0, ddof=1) / Xt_te.shape[0]
                    + np.var(Xc_te, axis=0, ddof=1) / Xc_te.shape[0]
                )
            )
            true_te = Xt_te.mean(0) - Xc_te.mean(0)
            true_rank = np.argsort(-np.abs(true_te))
            pred_rank = np.argsort(-np.abs(pred))
            keep = (pred - q * sigma_te > 0) | (pred + q * sigma_te < 0)
            kept_rank = np.argsort(-np.abs(pred[keep]))
            kept_global = np.arange(len(pred))[keep][kept_rank]
            prec_raw.append(precision_at_k(pred_rank, true_rank))
            prec_filt.append(precision_at_k(kept_global, true_rank))
    print("PRECISION_RAW", round(float(np.mean(prec_raw)), 4), flush=True)
    print("PRECISION_FILTERED", round(float(np.mean(prec_filt)), 4), flush=True)
    print("N", len(prec_raw), flush=True)


if __name__ == "__main__":
    main()
