import os
import sys

sys.path.insert(0, "/root/cw1ot")

import numpy as np
import anndata as ad

from w1ot.ot import w1ot

CONDS = [
    ("belinostat", "A549"),
    ("belinostat", "MCF7"),
    ("belinostat", "K562"),
    ("dacinostat", "A549"),
    ("dacinostat", "MCF7"),
    ("dacinostat", "K562"),
]
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


def delta_pred(Xc, Xt):
    return Xt.mean(0) - Xc.mean(0)


def w1ot_pred(Xc_tr, Xt_tr, Xc_eval, out):
    model = w1ot(Xc_tr, Xt_tr, device="cuda", path=out)
    model.fit_potential_function(num_iters=2500)
    model.fit_distance_function(num_iters=2500)
    transported = model.transport(Xc_eval)
    return transported.mean(0) - Xc_eval.mean(0)


def evaluate(pred_tr, pred_cal, pred_te, Xc_cal, Xt_cal, Xc_te, Xt_te):
    sigma_cal = shrink(
        np.sqrt(
            np.var(Xt_cal, axis=0, ddof=1) / Xt_cal.shape[0]
            + np.var(Xc_cal, axis=0, ddof=1) / Xc_cal.shape[0]
        )
    )
    true_cal = Xt_cal.mean(0) - Xc_cal.mean(0)
    scores = np.abs(true_cal - pred_cal) / (sigma_cal + 1e-6)
    q = quantile_higher(scores)
    sigma_te = shrink(
        np.sqrt(
            np.var(Xt_te, axis=0, ddof=1) / Xt_te.shape[0]
            + np.var(Xc_te, axis=0, ddof=1) / Xc_te.shape[0]
        )
    )
    true_te = Xt_te.mean(0) - Xc_te.mean(0)
    cover = np.abs(true_te - pred_te) <= q * sigma_te + 1e-6
    width = np.median(q * sigma_te)
    true_top = set(np.argsort(-np.abs(true_te))[:TOPK])
    pred_top = set(np.argsort(-np.abs(pred_te))[:TOPK])
    prec = len(true_top & pred_top) / TOPK
    return float(cover.mean()), float(width), prec


def main():
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")
    out_root = "/mnt/d/guo/CW1OT/results/uq_ablation"
    os.makedirs(out_root, exist_ok=True)
    res = {b: {"cov": [], "width": [], "prec": []} for b in ["delta", "w1ot"]}
    for drug, cell in CONDS:
        ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
        tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
        Xc_tr, Xc_cal, Xc_te = split_arrays(to_np(ctrl.X))
        Xt_tr, Xt_cal, Xt_te = split_arrays(to_np(tgt.X))
        d_tr = delta_pred(Xc_tr, Xt_tr)
        d_cal = d_tr
        d_te = d_tr
        cov, width, prec = evaluate(d_tr, d_cal, d_te, Xc_cal, Xt_cal, Xc_te, Xt_te)
        res["delta"]["cov"].append(cov)
        res["delta"]["width"].append(width)
        res["delta"]["prec"].append(prec)
        print(drug, cell, "delta", round(cov, 4), round(width, 5), round(prec, 4), flush=True)
        w_tr = w1ot_pred(Xc_tr, Xt_tr, Xc_tr, os.path.join(out_root, "w1ot"))
        w_cal = w1ot_pred(Xc_tr, Xt_tr, Xc_cal, os.path.join(out_root, "w1ot"))
        w_te = w1ot_pred(Xc_tr, Xt_tr, Xc_te, os.path.join(out_root, "w1ot"))
        cov, width, prec = evaluate(w_tr, w_cal, w_te, Xc_cal, Xt_cal, Xc_te, Xt_te)
        res["w1ot"]["cov"].append(cov)
        res["w1ot"]["width"].append(width)
        res["w1ot"]["prec"].append(prec)
        print(drug, cell, "w1ot", round(cov, 4), round(width, 5), round(prec, 4), flush=True)
    print("=== SUMMARY ===", flush=True)
    for b in ["delta", "w1ot"]:
        print(
            b,
            "cov",
            round(float(np.mean(res[b]["cov"])), 4),
            "width_med",
            round(float(np.median(res[b]["width"])), 5),
            "prec50",
            round(float(np.mean(res[b]["prec"])), 4),
            flush=True,
        )


if __name__ == "__main__":
    main()
