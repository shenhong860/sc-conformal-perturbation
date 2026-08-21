"""Aggregate scGPT results across training seeds (coverage, width, ensemble)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/mnt/d/guo/CW1OT/code")
from scgpt_finetune_norman import build_model, eval_perturb, load_pertdata  # noqa: E402

ROOT = Path("/mnt/d/guo/CW1OT/results/scgpt_norman")
SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.05
SHRINK = 0.5
OUT = ROOT / "multiseed.json"


def to_np(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


def shrink(sigma, lam=SHRINK):
    return lam * sigma + (1 - lam) * np.median(sigma)


def quantile_higher(scores, alpha=ALPHA):
    k = np.ceil((1 - alpha) * (1 + len(scores)))
    return np.sort(scores)[int(min(len(scores) - 1, k))]


def per_condition(res, ctrl_mean, conds):
    out = {}
    for cond in conds:
        idx = np.where(res["pert_cat"] == cond)[0]
        pred = res["pred"][idx]
        truth = res["truth"][idx]
        out[cond] = {
            "pred_logfc": pred.mean(0) - ctrl_mean,
            "true_logfc": truth.mean(0) - ctrl_mean,
            "n": len(idx),
            "var": truth.var(0, ddof=1),
        }
    return out


def evaluate(res, ctrl_mean, ctrl_var, n_ctrl, conds, q):
    cond_info = per_condition(res, ctrl_mean, conds)
    covs, widths, pears = [], [], []
    for d in cond_info.values():
        sigma = shrink(np.sqrt(d["var"] / d["n"] + ctrl_var / n_ctrl))
        cover = np.abs(d["true_logfc"] - d["pred_logfc"]) <= q * sigma + 1e-6
        covs.append(float(cover.mean()))
        widths.append(float(np.median(q * sigma)))
        pears.append(float(np.corrcoef(d["pred_logfc"], d["true_logfc"])[0, 1]))
    return {
        "coverage": float(np.mean(covs)),
        "width_median": float(np.median(widths)),
        "pearson": float(np.mean(pears)),
    }


def main():
    pert_data, _genes = load_pertdata()
    _m, _v, gene_ids = build_model(pert_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    ctrl_arr = to_np(ctrl_adata.X)
    ctrl_mean = ctrl_arr.mean(0)
    ctrl_var = ctrl_arr.var(0, ddof=1)
    n_ctrl = len(ctrl_adata)
    splits = json.loads((ROOT / "splits_union.json").read_text())
    val_loader = pert_data.dataloader["val_loader"]
    test_loader = pert_data.dataloader["test_loader"]

    per_seed = {}
    val_preds, test_preds = [], []
    test_truth = test_pert = None
    for seed in SEEDS:
        model, _v2, gids = build_model(pert_data)
        ckpt = ROOT / f"finetune_seed{seed}" / "best_model.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        model.to(device)
        val_res = eval_perturb(val_loader, model, gids)
        test_res = np.load(ROOT / f"finetune_seed{seed}" / "test_res.npz")
        val_preds.append(val_res["pred"])
        test_preds.append(test_res["pred"])
        if test_truth is None:
            test_truth = test_res["truth"]
            test_pert = test_res["pert_cat"]

        val_cond = per_condition(val_res, ctrl_mean, splits["val"])
        cal_scores = []
        for d in val_cond.values():
            sigma = shrink(np.sqrt(d["var"] / d["n"] + ctrl_var / n_ctrl))
            cal_scores.append(np.abs(d["true_logfc"] - d["pred_logfc"]) / (sigma + 1e-6))
        q = quantile_higher(np.concatenate(cal_scores))
        res = evaluate(test_res, ctrl_mean, ctrl_var, n_ctrl, splits["test"], q)
        per_seed[seed] = {"q": float(q), **res}
        print(f"seed {seed}: {res}", flush=True)

    cov = np.array([per_seed[s]["coverage"] for s in SEEDS])
    pear = np.array([per_seed[s]["pearson"] for s in SEEDS])
    width = np.array([per_seed[s]["width_median"] for s in SEEDS])

    # Ensemble across seeds.
    val_ens = {
        "pert_cat": val_res["pert_cat"],
        "pred": np.mean(val_preds, axis=0),
        "truth": val_res["truth"],
    }
    val_cond = per_condition(val_ens, ctrl_mean, splits["val"])
    cal_scores = []
    for d in val_cond.values():
        sigma = shrink(np.sqrt(d["var"] / d["n"] + ctrl_var / n_ctrl))
        cal_scores.append(np.abs(d["true_logfc"] - d["pred_logfc"]) / (sigma + 1e-6))
    q_ens = quantile_higher(np.concatenate(cal_scores))
    test_ens = {
        "pert_cat": test_pert,
        "pred": np.mean(test_preds, axis=0),
        "truth": test_truth,
    }
    ens = evaluate(test_ens, ctrl_mean, ctrl_var, n_ctrl, splits["test"], q_ens)
    print("ensemble:", ens, flush=True)

    def ci(x):
        return float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))

    result = {
        "seeds": SEEDS,
        "per_seed": per_seed,
        "coverage_mean": float(cov.mean()),
        "coverage_ci95": ci(cov),
        "coverage_per_seed": [round(float(x), 4) for x in cov],
        "width_median_mean": float(width.mean()),
        "width_median_per_seed": [round(float(x), 4) for x in width],
        "pearson_mean": float(pear.mean()),
        "pearson_per_seed": [round(float(x), 4) for x in pear],
        "ensemble": ens,
        "ensemble_q": float(q_ens),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
