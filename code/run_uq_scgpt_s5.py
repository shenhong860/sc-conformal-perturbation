"""Conformal evaluation with the fine-tuned scGPT base predictor on Norman."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/mnt/d/guo/CW1OT/code")
from scgpt_finetune_norman import (  # noqa: E402
    SAVE_DIR,
    WEIGHTS,
    build_model,
    eval_perturb,
    load_pertdata,
)

OUT = Path("/mnt/d/guo/CW1OT/results/scgpt_norman/conformal.json")
ALPHA = 0.05
SHRINK = 0.5


def to_np(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def quantile_higher(scores, alpha=ALPHA):
    k = np.ceil((1 - alpha) * (1 + len(scores)))
    return np.sort(scores)[int(min(len(scores) - 1, k))]


def shrink(sigma, lam=SHRINK):
    return lam * sigma + (1 - lam) * np.median(sigma)


def mean_std_cells(mat):
    arr = to_np(mat)
    return arr.mean(0), arr.std(0, ddof=1) ** 2 * (len(arr) - 1) / len(arr)


def per_condition_logfc(res, ctrl_mean, conds):
    out = {}
    for cond in conds:
        idx = np.where(res["pert_cat"] == cond)[0]
        pred = res["pred"][idx]
        truth = res["truth"][idx]
        out[cond] = {
            "pred_logfc": pred.mean(0) - ctrl_mean,
            "true_logfc": truth.mean(0) - ctrl_mean,
            "n": len(idx),
            "truth_var": truth.var(0, ddof=1),
        }
    return out


def split_arrays(arr, fracs=(0.5, 0.25, 0.25), seed=0):
    rng = np.random.RandomState(seed)
    n = len(arr)
    idx = rng.permutation(n)
    a = int(fracs[0] * n)
    b = a + int(fracs[1] * n)
    return arr[idx[:a]], arr[idx[a:b]], arr[idx[b:]]


def main():
    pert_data, genes = load_pertdata()
    model, _vocab, gene_ids = build_model(pert_data)
    model.load_state_dict(
        torch.load(SAVE_DIR / "best_model.pt", map_location="cpu", weights_only=False)
    )
    model.to(next(model.parameters()).device)

    val_res = eval_perturb(pert_data.dataloader["val_loader"], model, gene_ids)
    test_res = eval_perturb(pert_data.dataloader["test_loader"], model, gene_ids)

    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    ctrl_arr = to_np(ctrl_adata.X)
    ctrl_mean = ctrl_arr.mean(0)
    ctrl_var = ctrl_arr.var(0, ddof=1)
    n_ctrl = len(ctrl_adata)

    splits = json.loads(Path("/mnt/d/guo/CW1OT/results/scgpt_norman/splits_union.json").read_text())
    val_cond = per_condition_logfc(val_res, ctrl_mean, splits["val"])
    test_cond = per_condition_logfc(test_res, ctrl_mean, splits["test"])

    cal_scores = []
    for cond, d in val_cond.items():
        sigma = shrink(np.sqrt(d["truth_var"] / d["n"] + ctrl_var / n_ctrl))
        cal_scores.append(np.abs(d["true_logfc"] - d["pred_logfc"]) / (sigma + 1e-6))
    q = quantile_higher(np.concatenate(cal_scores))
    print("scGPT conformal quantile", round(float(q), 3), flush=True)

    scgpt_cov, scgpt_width = {}, {}
    for cond, d in test_cond.items():
        sigma = shrink(np.sqrt(d["truth_var"] / d["n"] + ctrl_var / n_ctrl))
        cover = np.abs(d["true_logfc"] - d["pred_logfc"]) <= q * sigma + 1e-6
        scgpt_cov[cond] = float(cover.mean())
        scgpt_width[cond] = float(np.median(q * sigma))
    print("scGPT overall coverage", round(float(np.mean(list(scgpt_cov.values()))), 4), flush=True)

    # Delta baseline on the same test conditions (S5 protocol, per-condition calibration).
    delta_cov, delta_width = {}, {}
    rng = np.random.RandomState(0)
    adata = pert_data.adata
    for cond in splits["test"]:
        tgt = adata[adata.obs["condition"] == cond]
        n = min(len(tgt), len(ctrl_adata))
        ci = rng.permutation(len(ctrl_adata))[:n]
        ctrl_s = ctrl_arr[ci]
        Xt = to_np(tgt.X)
        Xc_tr, Xc_cal, Xc_te = split_arrays(ctrl_s)
        Xt_tr, Xt_cal, Xt_te = split_arrays(Xt)
        pred = Xt_tr.mean(0) - Xc_tr.mean(0)
        sigma_cal = shrink(
            np.sqrt(
                Xt_cal.var(0, ddof=1) / Xt_cal.shape[0]
                + Xc_cal.var(0, ddof=1) / Xc_cal.shape[0]
            )
        )
        scores = np.abs((Xt_cal.mean(0) - Xc_cal.mean(0)) - pred) / (sigma_cal + 1e-6)
        qd = quantile_higher(scores)
        sigma_te = shrink(
            np.sqrt(
                Xt_te.var(0, ddof=1) / Xt_te.shape[0]
                + Xc_te.var(0, ddof=1) / Xc_te.shape[0]
            )
        )
        true_te = Xt_te.mean(0) - Xc_te.mean(0)
        delta_cov[cond] = float((np.abs(true_te - pred) <= qd * sigma_te + 1e-6).mean())
        delta_width[cond] = float(np.median(qd * sigma_te))
    print("delta overall coverage", round(float(np.mean(list(delta_cov.values()))), 4), flush=True)

    pearson = np.mean(
        [
            float(np.corrcoef(d["pred_logfc"], d["true_logfc"])[0, 1])
            for d in test_cond.values()
        ]
    )
    result = {
        "q": float(q),
        "scgpt_overall": float(np.mean(list(scgpt_cov.values()))),
        "delta_overall": float(np.mean(list(delta_cov.values()))),
        "scgpt_pearson": float(pearson),
        "scgpt_coverage": scgpt_cov,
        "delta_coverage": delta_cov,
        "scgpt_width": scgpt_width,
        "delta_width": delta_width,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
