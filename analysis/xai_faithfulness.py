"""
Core methodological experiment: can explainable ML (TreeSHAP over a model
trained only on indicator scores -> standing) faithfully recover the TRUE
drivers of a composite ranking, and detect a methodology change, WITHOUT
being told the weights?

Ground truth is available because the QS overall score is an exact linear
weighted sum, so the exact Shapley value of indicator k for institution i is
    phi_k(i) = w_k * (x_ik - mean_k).
We compare TreeSHAP from a model that never sees the weights against this.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import shap

from qs_common import (build_panel, weighted_score, W_OLD, W_NEW, IND, COMMON,
                       IND_LABEL)

warnings.filterwarnings("ignore")


def exact_linear_shap(X, weights, keys):
    """Closed-form Shapley values for f(x)=sum_k w_k x_k: phi_k = w_k (x_k - mean_k)."""
    mu = X.mean(axis=0)
    return (X - mu) * np.array([weights[k] for k in keys])


def recovered_weight(shap_k, x_k):
    """OLS slope (no intercept) of attribution_k on centered x_k ~= effective weight."""
    xc = x_k - x_k.mean()
    return float(np.dot(xc, shap_k) / np.dot(xc, xc))


def attributions(X, y, keys, weights, seed=0):
    """Return per-method SHAP-style attribution matrices for one regime."""
    out = {}
    out["exact"] = exact_linear_shap(X, weights, keys)               # oracle

    lin = LinearRegression().fit(X, y)
    out["linear"] = shap.LinearExplainer(lin, X).shap_values(X)      # correct model class

    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                               random_state=seed, n_jobs=-1).fit(X, y)
    out["_rf_r2"] = rf.score(X, y)
    out["tree_path"] = shap.TreeExplainer(
        rf, feature_perturbation="tree_path_dependent").shap_values(X)
    bg = shap.sample(X, 200, random_state=seed)
    # check_additivity disabled: interventional TreeSHAP with a finite background
    # sample can violate the additivity assertion by a tiny numerical margin on
    # rare samples; the attributions are unaffected (the 2024 regime passes).
    out["tree_interv"] = shap.TreeExplainer(
        rf, data=bg, feature_perturbation="interventional").shap_values(X, check_additivity=False)
    return out


def weight_table(X, attrs, keys, weights):
    rows = []
    for j, k in enumerate(keys):
        row = {"indicator": IND_LABEL[k], "true_w": weights[k]}
        for m in ("exact", "linear", "tree_path", "tree_interv"):
            row[m] = recovered_weight(attrs[m][:, j], X[:, j])
        rows.append(row)
    return pd.DataFrame(rows)


def run_regime(have, year, label):
    """Faithfulness benchmark for one edition's indicator scores under W_NEW."""
    X = have[[f"{k}_{year}" for k in IND]].values.astype(float)
    df = pd.DataFrame({k: have[f"{k}_{year}"].values for k in IND})
    y = weighted_score(df, W_NEW, keys=IND)

    C = np.corrcoef(X, rowvar=False)
    ar_corr_others = np.delete(np.abs(C[IND.index("ar")]), IND.index("ar"))
    print(f"\n========== {label} regime (n={len(have)}) ==========")
    print(f"Feature correlation: Academic Reputation mean |corr| = "
          f"{ar_corr_others.mean():.2f} (max {ar_corr_others.max():.2f}); "
          f"correlation-matrix condition number = {np.linalg.cond(C):.1f}")

    attrs = attributions(X, y, IND, W_NEW)
    print(f"RandomForest fit R^2 = {attrs['_rf_r2']:.4f}")
    wt = weight_table(X, attrs, IND, W_NEW)
    pd.set_option("display.width", 140)
    print(f"\n=== Recovered effective weights by attribution method ({label}) ===")
    print(wt.to_string(index=False,
          formatters={c: "{:.3f}".format for c in
                      ["true_w", "exact", "linear", "tree_path", "tree_interv"]}))
    print("\n=== Weight-recovery fidelity (mean abs error vs true weights) ===")
    for m in ("exact", "linear", "tree_path", "tree_interv"):
        mae = np.mean(np.abs(wt[m] - wt["true_w"]))
        corrs = [pearsonr(attrs[m][:, j], attrs["exact"][:, j])[0]
                 for j in range(len(IND)) if attrs[m][:, j].std() > 1e-9]
        print(f"  {m:12} MAE(weights)={mae:.3f}   mean corr-to-exact (instance level)="
              f"{np.mean(corrs):.3f}")
    path = f"output/faithfulness_weights_{year}.csv"
    wt.to_csv(path, index=False)
    print(f"Saved {path}")
    return wt


def robustness(have, year, n_seeds=20):
    """Sensitivity of default (path-dependent) TreeSHAP weight recovery to the
    random seed and to the model family: random forest vs gradient boosting.
    Returns a long DataFrame of recovered weights per (seed, model, indicator)."""
    X = have[[f"{k}_{year}" for k in IND]].values.astype(float)
    df = pd.DataFrame({k: have[f"{k}_{year}"].values for k in IND})
    y = weighted_score(df, W_NEW, keys=IND)
    rows = []
    for s in range(n_seeds):
        models = {
            "Random forest": RandomForestRegressor(
                n_estimators=400, min_samples_leaf=2, random_state=s, n_jobs=-1),
            "Gradient boosting": GradientBoostingRegressor(
                n_estimators=300, max_depth=3, random_state=s),
        }
        for mname, model in models.items():
            model.fit(X, y)
            r2 = model.score(X, y)
            # path-dependent TreeSHAP; check_additivity disabled so the 20-seed,
            # two-family sweep cannot abort on a sub-1% numerical tolerance.
            phi = shap.TreeExplainer(
                model, feature_perturbation="tree_path_dependent"
            ).shap_values(X, check_additivity=False)
            for j, k in enumerate(IND):
                rows.append({"seed": s, "model": mname, "r2": r2,
                             "indicator": IND_LABEL[k], "true": W_NEW[k],
                             "recovered": recovered_weight(phi[:, j], X[:, j])})
    return pd.DataFrame(rows)


def summarize_robustness(rob):
    """Per (model, indicator): mean and 95% CI of recovered weight, IND order."""
    order = [IND_LABEL[k] for k in IND]
    g = (rob.groupby(["model", "indicator"])
            .agg(mean=("recovered", "mean"),
                 ci_lo=("recovered", lambda v: v.quantile(.025)),
                 ci_hi=("recovered", lambda v: v.quantile(.975)),
                 true=("true", "first"))
            .reset_index())
    g["indicator"] = pd.Categorical(g["indicator"], categories=order, ordered=True)
    return g.sort_values(["model", "indicator"]).reset_index(drop=True)


def main():
    p = build_panel()
    # 2024 regime: same complete-data sample as the decomposition (matches Table 3)
    have24 = p.dropna(subset=[f"{k}_2023" for k in COMMON] +
                            [f"{k}_2024" for k in COMMON] + ["sus_2024"]).copy()
    run_regime(have24, 2024, "2024")
    # 2025 regime: replication on an independent edition (same methodology)
    have25 = p.dropna(subset=[f"{k}_2025" for k in IND]).copy()
    run_regime(have25, 2025, "2025 (replication)")

    # ---- robustness: random-seed and model-family sensitivity (2024 regime) ----
    print("\n========== Robustness: 20 seeds x {Random forest, Gradient boosting} (2024) ==========")
    rob = robustness(have24, 2024, n_seeds=20)
    rob.to_csv("output/robustness_raw.csv", index=False)
    summarize_robustness(rob).to_csv("output/robustness_summary.csv", index=False)
    newlab = [IND_LABEL[k] for k in ("irn", "ger", "sus")]
    for mname in ("Random forest", "Gradient boosting"):
        sub = rob[rob["model"] == mname]
        r2m = sub.groupby("seed")["r2"].first().mean()
        ar = sub[sub["indicator"] == IND_LABEL["ar"]]["recovered"]
        mae = sub.groupby("seed").apply(
            lambda d: (d["recovered"] - d["true"]).abs().mean(), include_groups=False)
        newmax = sub[sub["indicator"].isin(newlab)]["recovered"].max()
        print(f"  {mname:18} fit R2={r2m:.4f} | AR={ar.mean():.3f} "
              f"(95% CI {ar.quantile(.025):.3f}-{ar.quantile(.975):.3f}) | "
              f"new-indicator max={newmax:.3f} | "
              f"MAE={mae.mean():.3f} (sd {mae.std():.3f})")
    print("Saved output/robustness_raw.csv and output/robustness_summary.csv")


if __name__ == "__main__":
    main()
