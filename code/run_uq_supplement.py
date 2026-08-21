"""Multi-seed supplements for the UQ manuscript.

Run in WSL with the cw1ot conda environment.
Outputs human-readable log plus results/uq_supplement/results.json.

Added evidence:
- S3 leave-one-drug-out across 5 seeds with 95% CIs
- S3 baselines: no-shrinkage conformal, fixed-width normal interval
- S4 DEG use case across 5 seeds (flagged top-50 precision is the reported metric)
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np

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
TOPK = 50
SEEDS = [0, 1, 2, 3, 4]
Z_NORM = 1.959963984540054
OUT_DIR = Path("/mnt/d/guo/CW1OT/results/uq_supplement")


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


def se_calc(var, n):
    return np.sqrt(var / max(1, n))


def bootstrap_se(Xt, Xc, n_boot=200, seed=0):
    rng = np.random.RandomState(seed * 7919 + 13)
    nt, nc = len(Xt), len(Xc)
    boot = np.empty((n_boot, Xt.shape[1]), dtype=np.float64)
    for b in range(n_boot):
        it = rng.randint(0, nt, nt)
        ic = rng.randint(0, nc, nc)
        boot[b] = Xt[it].mean(0) - Xc[ic].mean(0)
    return boot.std(axis=0, ddof=1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad("/mnt/d/guo/CW1OT/data/sciplex3/hvg.h5ad")

    # Cache condition matrices once.
    cond_arrays = {}
    for drug in DRUGS:
        for cell in CELL_TYPES:
            ctrl = adata[(adata.obs["drug"] == "control") & (adata.obs["cell_type"] == cell)]
            tgt = adata[(adata.obs["drug"] == drug) & (adata.obs["cell_type"] == cell)]
            if len(ctrl) < 600 or len(tgt) < 600:
                continue
            cond_arrays[(drug, cell)] = (to_np(ctrl.X), to_np(tgt.X))

    # Per-seed accumulators.
    s3_seed = {key: [] for key in ("cp_shrink", "cp_noshrink", "fixed_shrink", "boot_gauss")}
    s3_cell_seed = {key: {c: [] for c in CELL_TYPES} for key in s3_seed}
    s3_width_seed = {key: [] for key in ("cp_shrink", "cp_noshrink", "fixed_shrink", "boot_gauss")}
    s2_seed = {key: [] for key in ("cp_shrink", "cp_noshrink", "fixed_shrink", "boot_gauss")}
    s2_width_seed = {key: [] for key in ("cp_shrink", "cp_noshrink", "fixed_shrink", "boot_gauss")}
    s4_seed = {key: [] for key in ("raw_prec50", "flag_prec50", "flag_recall", "frac_flagged")}
    s4_cond_count = []

    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        cond_info = {}
        for key, (Xc, Xt) in cond_arrays.items():
            Xc_tr, Xc_cal, Xc_te = split_arrays(Xc, seed=seed)
            Xt_tr, Xt_cal, Xt_te = split_arrays(Xt, seed=seed)
            pred_in = Xt_tr.mean(0) - Xc_tr.mean(0)
            true_cal = Xt_cal.mean(0) - Xc_cal.mean(0)
            true_te = Xt_te.mean(0) - Xc_te.mean(0)
            sigma_cal_raw = np.sqrt(
                np.var(Xt_cal, axis=0, ddof=1) / Xt_cal.shape[0]
                + np.var(Xc_cal, axis=0, ddof=1) / Xc_cal.shape[0]
            )
            sigma_te_raw = np.sqrt(
                np.var(Xt_te, axis=0, ddof=1) / Xt_te.shape[0]
                + np.var(Xc_te, axis=0, ddof=1) / Xc_te.shape[0]
            )
            sigma_te_boot = bootstrap_se(Xt_te, Xc_te, seed=seed)
            err_cal = np.abs(true_cal - pred_in)
            cond_info[key] = {
                "pred_in": pred_in,
                "true_te": true_te,
                "err_cal": err_cal,
                "sigma_cal_raw": sigma_cal_raw,
                "sigma_te_raw": sigma_te_raw,
                "sigma_te_boot": sigma_te_boot,
            }

        # ---- S3: leave-one-drug-out across variants ----
        for variant, lam, mode in (
            ("cp_shrink", 0.5, "conformal"),
            ("cp_noshrink", 1.0, "conformal"),
            ("fixed_shrink", 0.5, "fixed"),
            ("boot_gauss", None, "bootstrap"),
        ):
            cover_rows = []
            widths = []
            for held in DRUGS:
                for cell in CELL_TYPES:
                    test_keys = [k for k in cond_info if k[0] == held and k[1] == cell]
                    cal_keys = [k for k in cond_info if k[0] != held and k[1] == cell]
                    if not test_keys or not cal_keys:
                        continue
                    if mode == "bootstrap":
                        q = Z_NORM
                    elif mode == "conformal":
                        scores = []
                        for k in cal_keys:
                            info = cond_info[k]
                            sigma_cal = shrink(info["sigma_cal_raw"], lam)
                            scores.append(info["err_cal"] / (sigma_cal + 1e-6))
                        q = quantile_higher(np.concatenate(scores))
                    else:
                        q = Z_NORM
                    pred_out = np.mean([cond_info[k]["pred_in"] for k in cal_keys], axis=0)
                    for k in test_keys:
                        info = cond_info[k]
                        if mode == "bootstrap":
                            sigma_te = info["sigma_te_boot"]
                        else:
                            sigma_te = shrink(info["sigma_te_raw"], lam)
                        cover = np.abs(info["true_te"] - pred_out) <= q * sigma_te + 1e-6
                        cover_rows.append((k[1], float(cover.mean())))
                        widths.append(np.median(q * sigma_te))
            arr = np.array([r[1] for r in cover_rows])
            s3_seed[variant].append(float(arr.mean()))
            if variant in s3_width_seed:
                s3_width_seed[variant].append(float(np.mean(widths)))
            for cell in CELL_TYPES:
                c = np.array([r[1] for r in cover_rows if r[0] == cell])
                s3_cell_seed[variant][cell].append(float(c.mean()))
            print(
                f"S3 {variant}: overall={arr.mean():.4f} "
                + " ".join(f"{c}={np.mean([r[1] for r in cover_rows if r[0]==c]):.4f}" for c in CELL_TYPES),
                flush=True,
            )

        # ---- S4: DEG use case (flagged top-50 precision is the headline metric) ----
        raw_prec, flag_prec50, flag_recall, frac_flag = [], [], [], []
        for key, (Xc, Xt) in cond_arrays.items():
            Xc_tr, Xc_cal, Xc_te = split_arrays(Xc, seed=seed)
            Xt_tr, Xt_cal, Xt_te = split_arrays(Xt, seed=seed)
            pred = Xt_tr.mean(0) - Xc_tr.mean(0)
            sigma_cal = shrink(
                np.sqrt(
                    np.var(Xt_cal, axis=0, ddof=1) / Xt_cal.shape[0]
                    + np.var(Xc_cal, axis=0, ddof=1) / Xc_cal.shape[0]
                ),
                0.5,
            )
            scores = np.abs((Xt_cal.mean(0) - Xc_cal.mean(0)) - pred) / (sigma_cal + 1e-6)
            q = quantile_higher(scores)
            sigma_te = shrink(
                np.sqrt(
                    np.var(Xt_te, axis=0, ddof=1) / Xt_te.shape[0]
                    + np.var(Xc_te, axis=0, ddof=1) / Xc_te.shape[0]
                ),
                0.5,
            )
            true_te = Xt_te.mean(0) - Xc_te.mean(0)
            true_top = set(np.argsort(-np.abs(true_te))[:TOPK])
            pred_rank = np.argsort(-np.abs(pred))
            raw_prec.append(len(set(pred_rank[:TOPK]) & true_top) / TOPK)
            flagged = np.where(np.abs(pred) > q * sigma_te)[0]
            frac_flag.append(len(flagged) / len(pred))
            flag_recall.append(len(set(flagged.tolist()) & true_top) / TOPK)
            if len(flagged) >= TOPK:
                f50 = flagged[np.argsort(-np.abs(pred[flagged]))[:TOPK]]
                flag_prec50.append(len(set(f50.tolist()) & true_top) / TOPK)
        s4_seed["raw_prec50"].append(float(np.mean(raw_prec)))
        s4_seed["flag_prec50"].append(float(np.mean(flag_prec50)))
        s4_seed["flag_recall"].append(float(np.mean(flag_recall)))
        s4_seed["frac_flagged"].append(float(np.mean(frac_flag)))
        s4_cond_count.append(len(flag_prec50))
        print(
            f"S4: raw50={np.mean(raw_prec):.4f} flag50={np.mean(flag_prec50):.4f} "
            f"recall={np.mean(flag_recall):.4f} flagged_frac={np.mean(frac_flag):.4f} n={len(flag_prec50)}",
            flush=True,
        )

        # ---- S2: in-distribution group calibration across variants ----
        for variant, lam, mode in (
            ("cp_shrink", 0.5, "conformal"),
            ("cp_noshrink", 1.0, "conformal"),
            ("fixed_shrink", 0.5, "fixed"),
            ("boot_gauss", None, "bootstrap"),
        ):
            cover_rows = []
            widths = []
            for cell in CELL_TYPES:
                keys = [k for k in cond_info if k[1] == cell]
                if not keys:
                    continue
                if mode == "bootstrap":
                    q = Z_NORM
                elif mode == "conformal":
                    scores = []
                    for k in keys:
                        info = cond_info[k]
                        sigma_cal = shrink(info["sigma_cal_raw"], lam)
                        scores.append(info["err_cal"] / (sigma_cal + 1e-6))
                    q = quantile_higher(np.concatenate(scores))
                else:
                    q = Z_NORM
                for k in keys:
                    info = cond_info[k]
                    if mode == "bootstrap":
                        sigma_te = info["sigma_te_boot"]
                    else:
                        sigma_te = shrink(info["sigma_te_raw"], lam)
                    cover = np.abs(info["true_te"] - info["pred_in"]) <= q * sigma_te + 1e-6
                    cover_rows.append(float(cover.mean()))
                    widths.append(np.median(q * sigma_te))
            arr = np.array(cover_rows)
            s2_seed[variant].append(float(arr.mean()))
            s2_width_seed[variant].append(float(np.mean(widths)))
            print(f"S2 {variant}: overall={arr.mean():.4f}", flush=True)

    def ci(vals):
        arr = np.asarray(vals, dtype=float)
        return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))

    results = {"seeds": SEEDS}
    print("\n===== MULTI-SEED SUMMARY (n=5) =====", flush=True)
    for variant in s3_seed:
        m, c = ci(s3_seed[variant])
        results[f"s3_{variant}"] = {
            "mean": round(m, 4),
            "ci95": round(c, 4),
            "per_seed": [round(x, 4) for x in s3_seed[variant]],
        }
        print(f"S3 {variant}: {m:.4f} +/- {c:.4f}  seeds={[round(x,4) for x in s3_seed[variant]]}", flush=True)
        for cell in CELL_TYPES:
            cm, cc = ci(s3_cell_seed[variant][cell])
            results[f"s3_{variant}_{cell}"] = {
                "mean": round(cm, 4),
                "ci95": round(cc, 4),
                "per_seed": [round(x, 4) for x in s3_cell_seed[variant][cell]],
            }
            print(f"  {cell}: {cm:.4f} +/- {cc:.4f}", flush=True)
    for variant in s3_width_seed:
        m, c = ci(s3_width_seed[variant])
        results[f"s3_width_{variant}"] = {"mean": round(m, 4), "ci95": round(c, 4)}
        print(f"S3 width {variant}: {m:.4f} +/- {c:.4f}", flush=True)
    for variant in s2_seed:
        m, c = ci(s2_seed[variant])
        results[f"s2_{variant}"] = {
            "mean": round(m, 4),
            "ci95": round(c, 4),
            "per_seed": [round(x, 4) for x in s2_seed[variant]],
        }
        print(f"S2 {variant}: {m:.4f} +/- {c:.4f}  seeds={[round(x,4) for x in s2_seed[variant]]}", flush=True)
    for variant in s2_width_seed:
        m, c = ci(s2_width_seed[variant])
        results[f"s2_width_{variant}"] = {"mean": round(m, 4), "ci95": round(c, 4)}
        print(f"S2 width {variant}: {m:.4f} +/- {c:.4f}", flush=True)
    for key in s4_seed:
        m, c = ci(s4_seed[key])
        results[f"s4_{key}"] = {
            "mean": round(m, 4),
            "ci95": round(c, 4),
            "per_seed": [round(x, 4) for x in s4_seed[key]],
        }
        print(f"S4 {key}: {m:.4f} +/- {c:.4f}  seeds={[round(x,4) for x in s4_seed[key]]}", flush=True)
    results["s4_cond_count_per_seed"] = s4_cond_count
    results["s4_cond_count_mean"] = float(np.mean(s4_cond_count))

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("\nwrote", OUT_DIR / "results.json", flush=True)


if __name__ == "__main__":
    main()
