"""Revision supplements: shrinkage sensitivity, S3b multi-seed, scGPT widths."""

import json
from pathlib import Path

import anndata as ad
import numpy as np

DRUGS = [
    "belinostat", "dacinostat", "givinostat", "hesperadin", "tanespimycin",
    "jnj_26854165", "tak_901", "flavopiridol_hcl", "alvespimycin_hcl",
]
CELLS = ["A549", "MCF7", "K562"]
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.05
ROOT = Path("/mnt/d/guo/CW1OT/results/scgpt_norman")


def to_np(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def split_arrays(arr, fracs=(0.5, 0.25, 0.25), seed=0):
    rng = np.random.RandomState(seed)
    n = len(arr)
    idx = rng.permutation(n)
    a = int(fracs[0] * n)
    b = a + int(fracs[1] * n)
    return arr[idx[:a]], arr[idx[a:b]], arr[idx[b:]]


def shrink(sigma, lam):
    return lam * sigma + (1 - lam) * np.median(sigma)


def quantile_higher(scores, alpha=ALPHA):
    k = np.ceil((1 - alpha) * (1 + len(scores)))
    return np.sort(scores)[int(min(len(scores) - 1, k))]


def s2_sensitivity():
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")
    cond = {}
    for drug in DRUGS:
        for cell in CELLS:
            ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
            tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
            if len(ctrl) < 600 or len(tgt) < 600:
                continue
            cond[(drug, cell)] = (to_np(ctrl.X), to_np(tgt.X))
    results = {lam: [] for lam in (0.0, 0.25, 0.75, 1.0)}
    for seed in SEEDS:
        info = {}
        for key, (Xc, Xt) in cond.items():
            Xc_tr, Xc_cal, Xc_te = split_arrays(Xc, seed=seed)
            Xt_tr, Xt_cal, Xt_te = split_arrays(Xt, seed=seed)
            pred = Xt_tr.mean(0) - Xc_tr.mean(0)
            info[key] = {
                "err": np.abs((Xt_cal.mean(0) - Xc_cal.mean(0)) - pred),
                "sig_cal": np.sqrt(
                    Xt_cal.var(0, ddof=1) / Xt_cal.shape[0]
                    + Xc_cal.var(0, ddof=1) / Xc_cal.shape[0]
                ),
                "true_te": Xt_te.mean(0) - Xc_te.mean(0),
                "pred": pred,
                "sig_te": np.sqrt(
                    Xt_te.var(0, ddof=1) / Xt_te.shape[0]
                    + Xc_te.var(0, ddof=1) / Xc_te.shape[0]
                ),
                "cell": key[1],
            }
        for lam in results:
            covers = []
            for cell in CELLS:
                keys = [k for k in info if k[1] == cell]
                scores = np.concatenate(
                    [info[k]["err"] / (shrink(info[k]["sig_cal"], lam) + 1e-6) for k in keys]
                )
                q = quantile_higher(scores)
                for k in keys:
                    d = info[k]
                    cover = np.abs(d["true_te"] - d["pred"]) <= q * shrink(d["sig_te"], lam) + 1e-6
                    covers.append(float(cover.mean()))
            results[lam].append(float(np.mean(covers)))
    for lam, vals in results.items():
        arr = np.array(vals)
        print(f"S2 lambda={lam}: {arr.mean():.4f} +/- {1.96*arr.std(ddof=1)/np.sqrt(len(arr)):.4f}", flush=True)


def s3b_multiseed():
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")
    cond = {}
    for drug in DRUGS:
        for cell in CELLS:
            ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
            tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
            if len(ctrl) < 600 or len(tgt) < 600:
                continue
            cond[(drug, cell)] = (to_np(ctrl.X), to_np(tgt.X))
    for frac in (0.25, 0.10, 0.05):
        per_seed = []
        for seed in SEEDS:
            covers = []
            for key, (Xc, Xt) in cond.items():
                Xc_tr, Xc_cal, Xc_te = split_arrays(Xc, fracs=(0.5, frac, 0.5 - frac), seed=seed)
                Xt_tr, Xt_cal, Xt_te = split_arrays(Xt, fracs=(0.5, frac, 0.5 - frac), seed=seed)
                pred = Xt_tr.mean(0) - Xc_tr.mean(0)
                sig_cal = shrink(
                    np.sqrt(Xt_cal.var(0, ddof=1) / Xt_cal.shape[0] + Xc_cal.var(0, ddof=1) / Xc_cal.shape[0]), 0.5
                )
                scores = np.abs((Xt_cal.mean(0) - Xc_cal.mean(0)) - pred) / (sig_cal + 1e-6)
                q = quantile_higher(scores)
                sig_te = shrink(
                    np.sqrt(Xt_te.var(0, ddof=1) / Xt_te.shape[0] + Xc_te.var(0, ddof=1) / Xc_te.shape[0]), 0.5
                )
                true_te = Xt_te.mean(0) - Xc_te.mean(0)
                covers.append(float((np.abs(true_te - pred) <= q * sig_te + 1e-6).mean()))
            per_seed.append(float(np.mean(covers)))
        arr = np.array(per_seed)
        print(f"S3b frac={frac}: {arr.mean():.4f} +/- {1.96*arr.std(ddof=1)/np.sqrt(len(arr)):.4f} seeds={[round(x,4) for x in arr]}", flush=True)


def scgpt_width_percentiles():
    data = np.load(ROOT / "finetune" / "test_res.npz")
    ctrl_mean = np.load(ROOT / "finetune" / "ctrl_mean.npy")
    splits = json.loads((ROOT / "splits_union.json").read_text())
    conformal = json.loads((ROOT / "conformal.json").read_text())
    q = conformal["q"]
    ctrl_arr = None
    adata = ad.read_h5ad(ROOT / "data_union" / "norman_union" / "perturb_processed.h5ad")
    ctrl = adata[adata.obs["condition"] == "ctrl"]
    ctrl_arr = to_np(ctrl.X)
    ctrl_var = ctrl_arr.var(0, ddof=1)
    n_ctrl = len(ctrl)
    pcts = []
    for cond in splits["test"]:
        idx = np.where(data["pert_cat"] == cond)[0]
        truth = data["truth"][idx]
        sig = shrink(np.sqrt(truth.var(0, ddof=1) / len(idx) + ctrl_var / n_ctrl), 0.5)
        widths = q * sig
        pcts.append([np.percentile(widths, p) for p in (50, 90, 99)])
    arr = np.array(pcts)
    for j, p in enumerate((50, 90, 99)):
        print(f"scGPT width p{p}: median-of-conditions {np.median(arr[:, j]):.4f}, mean {arr[:, j].mean():.4f}", flush=True)


if __name__ == "__main__":
    s2_sensitivity()
    s3b_multiseed()
    scgpt_width_percentiles()
