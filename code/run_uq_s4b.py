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


def main():
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")
    raw_prec, flag_prec, flag_recall, frac_flag, flag_prec50 = [], [], [], [], []
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
            true_top = set(np.argsort(-np.abs(true_te))[:TOPK])
            pred_rank = np.argsort(-np.abs(pred))
            raw_prec.append(len(set(pred_rank[:TOPK]) & true_top) / TOPK)
            flagged = np.where(np.abs(pred) > q * sigma_te)[0]
            frac_flag.append(len(flagged) / len(pred))
            if len(flagged) > 0:
                flag_prec.append(len(set(flagged.tolist()) & true_top) / len(flagged))
            flag_recall.append(len(set(flagged.tolist()) & true_top) / TOPK)
            if len(flagged) >= TOPK:
                f50 = flagged[np.argsort(-np.abs(pred[flagged]))[:TOPK]]
                flag_prec50.append(len(set(f50.tolist()) & true_top) / TOPK)
    print("RAW_PREC50", round(float(np.mean(raw_prec)), 4), flush=True)
    print("FLAG_PREC", round(float(np.mean(flag_prec)), 4), flush=True)
    print("FLAG_RECALL", round(float(np.mean(flag_recall)), 4), flush=True)
    print("FRAC_FLAGGED", round(float(np.mean(frac_flag)), 4), flush=True)
    print("FLAG_PREC50", round(float(np.mean(flag_prec50)), 4) if flag_prec50 else "NA", "n", len(flag_prec50), flush=True)


if __name__ == "__main__":
    main()
