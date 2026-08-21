import os
import sys
import json
from pathlib import Path

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
CELLS = ["A549", "MCF7", "K562"]
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


def fit_linear_logfc(splits):
    """Joint per-gene linear model with drug and cell-line main effects.

    Trained on the train splits of all 24 conditions plus their controls.
    The predicted logFC of each drug is its main-effect coefficient, shared
    across cell lines, which is genuinely distinct from per-condition delta.
    """
    drugs = sorted({d for d, _ in CONDS})
    cells = CELLS
    drug_idx = {d: i for i, d in enumerate(drugs)}
    cell_idx = {c: i for i, c in enumerate(cells)}
    Zs, Ys = [], []
    for (d, c), (Xc_tr, Xt_tr, *_rest) in splits.items():
        for X, treated in ((Xc_tr, False), (Xt_tr, True)):
            n = len(X)
            z = np.zeros((n, 1 + len(drugs) + len(cells)), dtype=np.float32)
            z[:, 0] = 1.0
            z[:, 1 + len(drugs) + cell_idx[c]] = 1.0
            if treated:
                z[:, 1 + drug_idx[d]] = 1.0
            Zs.append(z)
            Ys.append(X)
    Z = np.vstack(Zs)
    Y = np.vstack(Ys)
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return {d: beta[1 + drug_idx[d]] for d in drugs}


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
    skip_w1ot = os.environ.get("SKIP_W1OT") == "1"
    res = {b: {"cov": [], "width": [], "prec": []} for b in ["delta", "linear", "w1ot"]}
    splits = {}
    for drug, cell in CONDS:
        ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
        tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
        Xc_tr, Xc_cal, Xc_te = split_arrays(to_np(ctrl.X))
        Xt_tr, Xt_cal, Xt_te = split_arrays(to_np(tgt.X))
        splits[(drug, cell)] = (Xc_tr, Xt_tr, Xc_cal, Xt_cal, Xc_te, Xt_te)

    for (drug, cell), (Xc_tr, Xt_tr, Xc_cal, Xt_cal, Xc_te, Xt_te) in splits.items():
        d_tr = delta_pred(Xc_tr, Xt_tr)
        cov, width, prec = evaluate(d_tr, d_tr, d_tr, Xc_cal, Xt_cal, Xc_te, Xt_te)
        res["delta"]["cov"].append(cov)
        res["delta"]["width"].append(width)
        res["delta"]["prec"].append(prec)
        print(drug, cell, "delta", round(cov, 4), round(width, 5), round(prec, 4), flush=True)

    betas = fit_linear_logfc(splits)
    for (drug, cell), (Xc_tr, Xt_tr, Xc_cal, Xt_cal, Xc_te, Xt_te) in splits.items():
        beta = betas[drug]
        cov, width, prec = evaluate(beta, beta, beta, Xc_cal, Xt_cal, Xc_te, Xt_te)
        res["linear"]["cov"].append(cov)
        res["linear"]["width"].append(width)
        res["linear"]["prec"].append(prec)
        print(drug, cell, "linear", round(cov, 4), round(width, 5), round(prec, 4), flush=True)

    if not skip_w1ot:
        for (drug, cell), (Xc_tr, Xt_tr, Xc_cal, Xt_cal, Xc_te, Xt_te) in splits.items():
            w_tr = w1ot_pred(Xc_tr, Xt_tr, Xc_tr, os.path.join(out_root, "w1ot"))
            w_cal = w1ot_pred(Xc_tr, Xt_tr, Xc_cal, os.path.join(out_root, "w1ot"))
            w_te = w1ot_pred(Xc_tr, Xt_tr, Xc_te, os.path.join(out_root, "w1ot"))
            cov, width, prec = evaluate(w_tr, w_cal, w_te, Xc_cal, Xt_cal, Xc_te, Xt_te)
            res["w1ot"]["cov"].append(cov)
            res["w1ot"]["width"].append(width)
            res["w1ot"]["prec"].append(prec)
            print(drug, cell, "w1ot", round(cov, 4), round(width, 5), round(prec, 4), flush=True)

    summary = {}
    for b in ["delta", "linear", "w1ot"]:
        if not res[b]["cov"]:
            continue
        summary[b] = {
            "coverage": float(np.mean(res[b]["cov"])),
            "width_median": float(np.median(res[b]["width"])),
            "precision50": float(np.mean(res[b]["prec"])),
        }
    Path(out_root, "ablation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=== SUMMARY ===", flush=True)
    for b, s in summary.items():
        print(
            b,
            "cov",
            round(s["coverage"], 4),
            "width_med",
            round(s["width_median"], 5),
            "prec50",
            round(s["precision50"], 4),
            flush=True,
        )


if __name__ == "__main__":
    main()
