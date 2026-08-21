import numpy as np


def simulate(n_gene=1200, n_cal=400, n_test=400, seed=0, n_groups=4):
    rng = np.random.RandomState(seed)
    group = rng.randint(0, n_groups, size=n_gene)
    true = rng.randn(n_gene) * 0.5
    base_sigma = 0.1 + 0.4 * (group / (n_groups - 1))
    pred = true + rng.randn(n_gene) * base_sigma
    idx = rng.permutation(n_gene)
    calib = idx[:n_cal]
    test = idx[n_cal : n_cal + n_test]

    sigma_hat = {}
    for g in range(n_groups):
        mask = group[calib] == g
        sigma_hat[g] = np.std(pred[calib][mask] - true[calib][mask]) + 1e-6 if mask.sum() > 0 else 1.0

    scores = np.abs(pred[calib] - true[calib]) / np.array([sigma_hat[group[i]] for i in calib])
    alpha = 0.05
    q = np.quantile(scores, min(1.0, np.ceil((1 - alpha) * (1 + len(scores))) / len(scores)))

    lo = pred[test] - q * np.array([sigma_hat[group[i]] for i in test])
    hi = pred[test] + q * np.array([sigma_hat[group[i]] for i in test])
    cover = (true[test] >= lo) & (true[test] <= hi)
    width = hi - lo
    return float(cover.mean()), float(width.mean()), cover, group[test]


def main():
    seeds = list(range(5))
    covs = []
    for seed in seeds:
        cov, w, cover, gtest = simulate(seed=seed)
        covs.append(cov)
        gcov = [float(cover[gtest == g].mean()) for g in range(4)]
        print("seed", seed, "coverage", round(cov, 4), "width", round(w, 4), "group_cov", [round(x, 3) for x in gcov], flush=True)
    arr = np.array(covs)
    print("MEAN_COVERAGE", round(float(arr.mean()), 4), "sd", round(float(arr.std()), 4), flush=True)


if __name__ == "__main__":
    main()
