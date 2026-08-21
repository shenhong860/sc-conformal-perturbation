import numpy as np
import anndata as ad

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
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/norman/norman_hvg1000.h5ad")
    guid = adata.obs["guide_identity"].values
    ctrl_mask = np.array(["NegCtrl" in g for g in guid])
    ctrl = adata[ctrl_mask]
    counts = {g: int(np.sum(guid == g)) for g in np.unique(guid) if "NegCtrl" not in g and np.sum(guid == g) >= 600}
    perts = sorted(counts, key=counts.get, reverse=True)[:30]
    rng = np.random.RandomState(0)
    covers = []
    for g in perts:
        tgt = adata[guid == g]
        n = min(len(tgt), len(ctrl))
        ci = rng.permutation(len(ctrl))[:n]
        ctrl_s = ctrl[ci]
        Xc_tr, Xc_cal, Xc_te = split_arrays(to_np(ctrl_s.X))
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
        cover = np.abs(true_te - pred) <= q * sigma_te + 1e-6
        covers.append(float(cover.mean()))
        print(g[:40], "n", len(tgt), "coverage", round(float(cover.mean()), 4), flush=True)
    arr = np.array(covers)
    print("OVERALL_COVERAGE", round(float(arr.mean()), 4), "n", len(arr), flush=True)


if __name__ == "__main__":
    main()
