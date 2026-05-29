"""
Substantive results + publication figures for the QS faithful-XAI study.

Produces:
  - score- and rank-level decomposition of the 2023->2024 reform into
    methodology vs performance components;
  - distributional incidence (who the reform advantaged) by region and
    research intensity;
  - a counterfactual reweighting simulator and its distributional read-out;
  - paper figures 1-5.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qs_common import (build_panel, decompose_2023_2024, decompose_2024_2025,
                       recover_weights, weighted_score, load_year, FILES,
                       W_OLD, W_NEW, IND, COMMON, IND_LABEL, OUT)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
                     "axes.grid": True, "grid.alpha": .3, "axes.axisbelow": True})
TEAL, CORAL, NAVY, GOLD = "#2a9d8f", "#e76f51", "#264653", "#e9c46a"
RES_LABEL = {"VH": "Very High", "HI": "High", "MD": "Medium", "LO": "Low"}


def panel_complete():
    p = build_panel()
    have = p.dropna(subset=[f"{k}_2023" for k in COMMON] +
                          [f"{k}_2024" for k in COMMON] + ["sus_2024"]).copy()
    return have


def rank_decomposition(have):
    """Telescoping rank decomposition: observed 2023->2024 rank change =
    methodology component + performance component (exact within the panel).
    Sustainability is unobserved in 2023; proxied by its 2024 value (stable)."""
    df23 = pd.DataFrame({k: have[f"{k}_2023"].values for k in IND})
    df23["sus"] = have["sus_2024"].values                  # proxy: stable sustainability
    df24 = pd.DataFrame({k: have[f"{k}_2024"].values for k in IND})

    s_old23 = weighted_score(df23, W_OLD, keys=IND)
    s_newrules23 = weighted_score(df23, W_NEW, keys=IND)   # 2023 perf, new rules
    s_new24 = weighted_score(df24, W_NEW, keys=IND)

    r_old23 = rankdata(-s_old23, method="average")
    r_newrules23 = rankdata(-s_newrules23, method="average")
    r_new24 = rankdata(-s_new24, method="average")

    out = have[["institution", "country", "region", "research"]].copy()
    out["rank_methodology"] = r_old23 - r_newrules23       # +ve = reform lifted them
    out["rank_performance"] = r_newrules23 - r_new24
    out["rank_observed"] = r_old23 - r_new24
    return out


def counterfactual_rank(have, weights):
    """Re-rank the 2024 cohort under an arbitrary weight vector (simulator)."""
    df24 = pd.DataFrame({k: have[f"{k}_2024"].values for k in IND})
    return rankdata(-weighted_score(df24, weights, keys=IND), method="average")


def main():
    have = panel_complete()
    dec = decompose_2023_2024(have)
    rd = rank_decomposition(have)
    n = len(have)
    print(f"n = {n} institutions\n")

    # ---- variance decomposition (score & rank) ----
    v_meth_s, v_perf_s = dec["methodology"].var(), dec["performance"].var()
    print(f"SCORE change: var(methodology)={v_meth_s:.2f}  var(performance)={v_perf_s:.2f}  "
          f"-> methodology = {v_meth_s/(v_meth_s+v_perf_s):.0%} of explained variance")
    v_meth_r, v_perf_r = rd["rank_methodology"].var(), rd["rank_performance"].var()
    print(f"RANK  change: var(methodology)={v_meth_r:.0f}  var(performance)={v_perf_r:.0f}  "
          f"-> methodology = {v_meth_r/(v_meth_r+v_perf_r):.0%} of explained variance")
    print(f"corr(observed rank change, methodology component) = "
          f"{pearsonr(rd['rank_observed'], rd['rank_methodology'])[0]:.2f}")

    # ---- distributional incidence by region ----
    print("\nMean methodology effect (score pts) by region:")
    reg = dec.groupby("region")["methodology"].agg(["mean", "count"]).query("count>=10")
    print(reg.sort_values("mean", ascending=False).round(2).to_string())

    print("\nMean methodology effect (score pts) by research intensity:")
    resd = dec.assign(research=have["research"].values)
    rr = resd.groupby("research")["methodology"].agg(["mean", "count"])
    rr = rr.reindex(["VH", "HI", "MD", "LO"]).dropna()
    print(rr.round(2).to_string())

    # ---- counterfactual simulator demo: same 2024 cohort, old vs new rules ----
    r_new = counterfactual_rank(have, W_NEW)
    r_old = counterfactual_rank(have, W_OLD)
    sim = have[["region"]].copy()
    sim["delta"] = r_old - r_new          # +ve = new rules improved their rank
    print("\nSimulator (2024 cohort, switching OLD->NEW rules) — mean rank gain by region:")
    print(sim.groupby("region")["delta"].mean().round(1).sort_values(ascending=False).to_string())

    # ---- per-edition weight recovery (RQ1) + reform-vs-placebo weight shift ----
    rec = {}
    for yr in (2023, 2024, 2025):
        w, r2, nrec = recover_weights(yr)
        rec[yr] = w
        print(f"\nRecovered weights {yr} (R^2={r2:.5f}, n={nrec}): " +
              " ".join(f"{k}={w[k]*100:.1f}" for k in IND))
    mae = lambda a, b: float(np.mean([abs(a[k] - b[k]) for k in IND]))
    print(f"\nRecovered-weight shift  2023->2024 (reform)  = {mae(rec[2023], rec[2024])*100:.2f} pp")
    print(f"Recovered-weight shift  2024->2025 (placebo) = {mae(rec[2024], rec[2025])*100:.2f} pp")
    pd.DataFrame(rec).T.to_csv(f"{OUT}/recovered_weights_by_year.csv")

    # ---- placebo: 2024->2025 (no reform) decomposition ----
    pl = build_panel().dropna(subset=[f"{k}_2024" for k in IND] +
                                     [f"{k}_2025" for k in IND]).copy()
    pdec = decompose_2024_2025(pl)
    vm, vp = pdec["methodology"].var(), pdec["performance"].var()
    print(f"\nPLACEBO 2024->2025 (n={len(pl)}): var(methodology)={vm:.4f} "
          f"var(performance)={vp:.2f} -> methodology = {vm/(vm+vp):.1%} of variance "
          f"(identity max|err|={pdec['identity_err'].abs().max():.1e})")
    pdec.to_csv(f"{OUT}/decomposition_2024_2025.csv", index=False)

    _figures(dec, rd, have, rec)
    graphical_abstract()
    graphical_abstract_a4()
    # save tables
    dec.to_csv(f"{OUT}/decomposition_scores.csv", index=False)
    rd.to_csv(f"{OUT}/decomposition_ranks.csv", index=False)
    print(f"\nFigures + tables written to {OUT}")


def _figures(dec, rd, have, rec=None):
    # Fig 1: data overview — what the three-year QS panel looks like
    frames = {y: load_year(y) for y in FILES}
    panel = build_panel()
    yrs = list(FILES)
    cols3 = [NAVY, TEAL, GOLD]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    # (a) cohort size per edition, with the matched panel marked
    sizes = [int(frames[y]["rank_num"].notna().sum()) for y in yrs]
    bars = ax[0].bar([str(y) for y in yrs], sizes, color=cols3)
    for b, s in zip(bars, sizes):
        ax[0].text(b.get_x() + b.get_width() / 2, s + 6, str(s), ha="center", fontsize=8)
    ax[0].axhline(len(panel), color=CORAL, ls="--", lw=1.3,
                  label=f"matched in all 3 ({len(panel)})")
    ax[0].set_ylim(0, max(sizes) * 1.13)
    ax[0].set_ylabel("Ranked institutions")
    ax[0].set_title("(a) Cohort size by edition")
    ax[0].legend(fontsize=8, loc="lower center")
    # (b) computed overall-score distribution by edition (matched panel)
    scores = []
    for y, w in ((2023, W_OLD), (2024, W_NEW), (2025, W_NEW)):
        d = pd.DataFrame({k: panel[f"{k}_{y}"].values for k in IND})
        s = weighted_score(d, w, keys=IND)
        scores.append(s[np.isfinite(s)])
    bp = ax[1].boxplot(scores, labels=[str(y) for y in yrs], patch_artist=True,
                       showfliers=False, medianprops={"color": "k"})
    for patch, c in zip(bp["boxes"], cols3):
        patch.set_facecolor(c); patch.set_alpha(.75)
    ax[1].set_ylabel("Computed overall score")
    ax[1].set_title("(b) Score distribution by edition")
    # (c) mean indicator score by edition (heatmap; Sustainability absent in 2023)
    M = np.array([[np.nanmean(panel[f"{k}_{y}"].values) for y in yrs] for k in IND])
    im = ax[2].imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=np.nanmax(M))
    ax[2].set_xticks(range(len(yrs))); ax[2].set_xticklabels([str(y) for y in yrs])
    ax[2].set_yticks(range(len(IND)))
    ax[2].set_yticklabels([IND_LABEL[k] for k in IND], fontsize=7.5)
    for i in range(len(IND)):
        for j in range(len(yrs)):
            v = M[i, j]
            label = "n/a" if np.isnan(v) else f"{v:.0f}"
            dark = (not np.isnan(v)) and v > np.nanmax(M) * 0.55
            ax[2].text(j, i, label, ha="center", va="center", fontsize=7,
                       color="white" if dark else "black")
    ax[2].set_title("(c) Mean indicator score by edition")
    fig.colorbar(im, ax=ax[2], fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig1_data_overview.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: score decomposition — methodology vs performance distribution
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(dec["methodology"], bins=40, color=TEAL, alpha=.8, label="Methodology")
    ax[0].hist(dec["performance"], bins=40, color=CORAL, alpha=.6, label="Performance")
    ax[0].set_xlabel("Contribution to 2023→2024 score change (pts)")
    ax[0].set_ylabel("Institutions"); ax[0].legend()
    ax[0].set_title("(a) Methodology dominates score change")
    ax[1].scatter(dec["performance"], dec["methodology"], s=9, alpha=.4, color=NAVY)
    ax[1].axhline(0, color="k", lw=.6); ax[1].axvline(0, color="k", lw=.6)
    ax[1].set_xlabel("Performance component (pts)")
    ax[1].set_ylabel("Methodology component (pts)")
    ax[1].set_title("(b) Per-institution components")
    fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig2_decomposition.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: rank-level decomposition by region (mean methodology vs performance)
    g = rd.groupby("region")[["rank_methodology", "rank_performance"]].mean()
    g = g.loc[g.index.isin(["Asia", "Europe", "Americas", "Oceania", "Africa"])]
    g = g.sort_values("rank_methodology")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(g))
    ax.barh(y - .2, g["rank_methodology"], .4, color=TEAL, label="Methodology")
    ax.barh(y + .2, g["rank_performance"], .4, color=CORAL, label="Performance")
    ax.set_yticks(y); ax.set_yticklabels(g.index)
    ax.axvline(0, color="k", lw=.7)
    ax.set_xlabel("Mean rank change within panel (+ = improved)")
    ax.set_title("Rank change decomposition by region (2023→2024)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig3_rank_by_region.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: distributional incidence by research intensity
    resd = dec.assign(research=have["research"].values)
    rr = resd.groupby("research")["methodology"].mean().reindex(["VH", "HI", "MD", "LO"]).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = [TEAL if v >= 0 else CORAL for v in rr.values]
    ax.bar([RES_LABEL[i] for i in rr.index], rr.values, color=colors)
    ax.axhline(0, color="k", lw=.7)
    ax.set_ylabel("Mean methodology effect (score pts)")
    ax.set_xlabel("Research intensity")
    ax.set_title("Who the reform advantaged, by research intensity")
    fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig4_research_intensity.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 5: counterfactual weight sweep — sustainability weight vs regional mean rank
    base_regions = ["Asia", "Europe", "Americas", "Oceania", "Africa"]
    sweep = np.linspace(0, 0.15, 16)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for reg, col in zip(base_regions, [TEAL, NAVY, CORAL, GOLD, "#8a5a44"]):
        means = []
        for s in sweep:
            w = dict(W_NEW); extra = s - W_NEW["sus"]
            w["sus"] = s; w["ar"] = W_NEW["ar"] - extra      # fund SUS from Academic Rep.
            r = counterfactual_rank(have, w)
            means.append(np.mean(r[have["region"].values == reg]))
        ax.plot(sweep * 100, means, marker="o", ms=3, color=col, label=reg)
    ax.axvline(5, color="k", ls="--", lw=.7, label="actual (5%)")
    ax.set_xlabel("Sustainability weight (%), funded from Academic Reputation")
    ax.set_ylabel("Mean panel rank (lower = better)")
    ax.invert_yaxis()
    ax.set_title("Counterfactual: regional standing vs sustainability weight")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig6_simulator.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 4: faithfulness — recovered weights by attribution method
    try:
        wt = pd.read_csv(f"{OUT}/faithfulness_weights_2024.csv")
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(wt)); bw = .2
        ax.bar(x - 1.5*bw, wt["true_w"], bw, label="True weight", color="k")
        ax.bar(x - .5*bw, wt["exact"], bw, label="Exact Shapley", color=TEAL)
        ax.bar(x + .5*bw, wt["tree_path"], bw, label="TreeSHAP (default)", color=CORAL)
        ax.bar(x + 1.5*bw, wt["tree_interv"], bw, label="TreeSHAP (interventional)", color=GOLD)
        ax.set_xticks(x); ax.set_xticklabels([s.replace(" ", "\n") for s in wt["indicator"]],
                                             fontsize=7.5)
        ax.set_ylabel("Recovered effective weight")
        ax.set_title("XAI weight recovery vs ground truth (2024 methodology)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig5_faithfulness.png", bbox_inches="tight")
        plt.close(fig)
    except FileNotFoundError:
        print("(run xai_faithfulness.py first for Fig 5)")

    # Fig 6: recovered weights by edition — reform jump (2023->2024) + placebo stability (2024->2025)
    if rec is not None:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(IND)); bw = .26
        for off, yr, col in [(-bw, 2023, NAVY), (0, 2024, TEAL), (bw, 2025, GOLD)]:
            ax.bar(x + off, [rec[yr][k] * 100 for k in IND], bw, label=str(yr), color=col)
        ax.set_xticks(x)
        ax.set_xticklabels([IND_LABEL[k].replace(" ", "\n") for k in IND], fontsize=7.5)
        ax.set_ylabel("Recovered weight (%)")
        ax.set_title("Recovered QS weights by edition: reform (2023→2024) vs no-reform (2024→2025)")
        ax.legend(title="Edition", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig7_weights_by_year.png", bbox_inches="tight")
        plt.close(fig)

    # Fig 7: faithfulness replication on the 2025 edition (mirror of Fig 5)
    try:
        wt = pd.read_csv(f"{OUT}/faithfulness_weights_2025.csv")
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(wt)); bw = .2
        ax.bar(x - 1.5*bw, wt["true_w"], bw, label="True weight", color="k")
        ax.bar(x - .5*bw, wt["exact"], bw, label="Exact Shapley", color=TEAL)
        ax.bar(x + .5*bw, wt["tree_path"], bw, label="TreeSHAP (default)", color=CORAL)
        ax.bar(x + 1.5*bw, wt["tree_interv"], bw, label="TreeSHAP (interventional)", color=GOLD)
        ax.set_xticks(x); ax.set_xticklabels([s.replace(" ", "\n") for s in wt["indicator"]],
                                             fontsize=7.5)
        ax.set_ylabel("Recovered effective weight")
        ax.set_title("XAI weight recovery vs ground truth (2025 edition: replication)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig8_faithfulness_2025.png", bbox_inches="tight")
        plt.close(fig)
    except FileNotFoundError:
        print("(run xai_faithfulness.py for Fig 7: 2025 faithfulness)")

    # Fig 8: robustness — recovered weights across 20 seeds x {RF, GBM} with 95% CI
    try:
        rs = pd.read_csv(f"{OUT}/robustness_summary.csv")
        order = [IND_LABEL[k] for k in IND]
        models = ["Random forest", "Gradient boosting"]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(order)); bw = .26
        true_w = rs[rs["model"] == models[0]].set_index("indicator").loc[order, "true"].values
        ax.bar(x - bw, true_w, bw, label="True weight", color="k")
        for off, m, col in [(0.0, models[0], CORAL), (bw, models[1], NAVY)]:
            d = rs[rs["model"] == m].set_index("indicator").loc[order]
            yerr = np.vstack([d["mean"] - d["ci_lo"], d["ci_hi"] - d["mean"]])
            ax.bar(x + off, d["mean"].values, bw, label=m, color=col,
                   yerr=yerr, capsize=2, error_kw={"lw": .8})
        ax.set_xticks(x); ax.set_xticklabels([s.replace(" ", "\n") for s in order], fontsize=7.5)
        ax.set_ylabel("Recovered effective weight")
        ax.set_title("Robustness of default TreeSHAP recovery (2024): 20 seeds, two model families")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/paper_fig9_robustness.png", bbox_inches="tight")
        plt.close(fig)
    except FileNotFoundError:
        print("(run xai_faithfulness.py for Fig 8: robustness)")


def graphical_abstract():
    """Standalone graphical abstract: true weights vs what default TreeSHAP reports
    for the reputation hub and the three reform indicators, plus the validation."""
    labels = ["Academic\nReputation", "Sustainability", "Employment\nOutcomes",
              "Intl Research\nNetwork"]
    true_w = [0.30, 0.05, 0.05, 0.05]
    shap_w = [0.65, 0.012, 0.016, 0.008]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(len(labels)); bw = 0.38
    ax.bar(x - bw / 2, true_w, bw, label="True weight (ground truth)", color=NAVY)
    ax.bar(x + bw / 2, shap_w, bw, label="What TreeSHAP reports", color=CORAL)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Effective weight in the QS score")
    ax.set_ylim(0, 0.82)
    ax.annotate("over-credited\nabout 2x", xy=(0 + bw / 2, 0.66), xytext=(0 + bw / 2, 0.71),
                ha="center", fontsize=9, color=CORAL, fontweight="bold")
    ax.annotate("the 3 indicators the reform added:\nunder-credited 5 to 6 times",
                xy=(2 + bw / 2, 0.03), xytext=(2.0, 0.30), ha="center", fontsize=9,
                color=CORAL, arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2))
    ax.set_title("When explanations mislead: auditing the 2024 QS ranking reform",
                 fontsize=13, fontweight="bold")
    ax.text(0.985, 0.97,
            "A high-accuracy model plus SHAP over-credits the correlated\n"
            "reputation indicator and hides what the reform actually rewarded.\n"
            "Ground truth comes from the exactly-linear QS score; the finding\n"
            "holds across 2023 to 2025, a no-reform placebo, and 20 seeds\n"
            "and two model families.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="#f4f1de", ec="#bbbbbb", alpha=.95))
    ax.legend(loc="upper center", fontsize=9, bbox_to_anchor=(0.34, 1.0))
    fig.tight_layout()
    fig.savefig(f"{OUT}/graphical_abstract.png", bbox_inches="tight")
    plt.close(fig)


def graphical_abstract_a4():
    """Full-page A4 (portrait) visual summary of the study, built from the
    manuscript's abstract: the key idea, the core result, the 79% decomposition,
    and the three validation pillars."""
    import textwrap
    BG = "#f4f1de"

    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
    fig.patch.set_facecolor("white")

    def panel(rect, fc="white", ec="#cccccc"):
        ax = fig.add_axes(rect); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=fc, ec=ec, lw=1.0))
        return ax

    # --- title ---
    fig.text(0.5, 0.978, "When Explanations Mislead", ha="center", va="top",
             fontsize=21, fontweight="bold", color=NAVY)
    fig.text(0.5, 0.949,
             "A ground-truth test of explainable machine learning for auditing university-ranking reforms",
             ha="center", va="top", fontsize=10.5, style="italic", color="#333333")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.935, 0.935], color=CORAL, lw=2.2,
                              transform=fig.transFigure))
    fig.text(0.5, 0.921,
             "The 2024 QS reform moved thousands of universities at once. Which factors truly drove each move?\n"
             "Panel of 1,390 universities matched across the 2023, 2024 and 2025 editions.",
             ha="center", va="top", fontsize=9.5, color="#222222")

    # --- Panel A: the key idea ---
    axA = panel([0.07, 0.812, 0.86, 0.078], fc=BG)
    axA.text(0.025, 0.86, "THE KEY IDEA", transform=axA.transAxes, fontsize=9.5,
             fontweight="bold", color=CORAL, va="top")
    axA.text(0.025, 0.60, textwrap.fill(
        "The QS overall score equals its indicator scores times their published weights, summed, almost "
        "exactly (reconstruction R-squared about 1.000 in 2023, 2024 and 2025). Because the true recipe is "
        "known, each indicator's exact contribution is available in closed form, so any explainer can be "
        "graded against ground truth.", width=112),
        transform=axA.transAxes, fontsize=8.8, va="top")

    # --- Panel B: the core result (bar chart) ---
    axB = fig.add_axes([0.11, 0.560, 0.82, 0.190])
    labels = ["Academic\nReputation", "Sustainability", "Employment\nOutcomes", "Intl Research\nNetwork"]
    true_w = [0.30, 0.05, 0.05, 0.05]; shap_w = [0.65, 0.012, 0.016, 0.008]
    x = np.arange(len(labels)); bw = 0.38
    axB.bar(x - bw / 2, true_w, bw, label="True weight (ground truth)", color=NAVY)
    axB.bar(x + bw / 2, shap_w, bw, label="What TreeSHAP reports", color=CORAL)
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=8.3)
    axB.set_ylim(0, 0.80); axB.set_ylabel("Weight in the QS score", fontsize=9)
    axB.set_title("What a popular explainer reports vs the truth (2024 weights)",
                  fontsize=11, fontweight="bold", color=NAVY, pad=8)
    axB.legend(loc="upper center", bbox_to_anchor=(0.60, 0.99), fontsize=8.3)
    axB.annotate("over-credited\nabout 2x", xy=(bw / 2, 0.66), xytext=(bw / 2, 0.72),
                 ha="center", fontsize=8.3, color=CORAL, fontweight="bold")
    axB.annotate("the 3 indicators the reform added:\nunder-credited 5 to 6 times",
                 xy=(2 + bw / 2, 0.03), xytext=(2.05, 0.30), ha="center", fontsize=8.3,
                 color=CORAL, arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.1))
    axB.grid(axis="y", alpha=.3); axB.set_axisbelow(True)

    # --- Panel C: the 79% decomposition ---
    fig.text(0.11, 0.512,
             "About four-fifths of the 2024 reshuffle came from the rule change, not from real improvement.",
             ha="left", va="bottom", fontsize=9, color="#222222")
    axC = fig.add_axes([0.11, 0.460, 0.82, 0.045])
    axC.barh([0], [79], color=NAVY); axC.barh([0], [21], left=[79], color=TEAL)
    axC.set_xlim(0, 100); axC.set_ylim(-0.5, 0.5); axC.axis("off")
    axC.text(39.5, 0, "Methodology / new rules   79%", ha="center", va="center",
             color="white", fontsize=9, fontweight="bold")
    axC.text(89.5, 0, "Performance   21%", ha="center", va="center", color="white", fontsize=8.5)

    # --- Panel D: three validation pillars ---
    dpos = [0.07, 0.3625, 0.655]; dw = 0.275
    dtitles = ["GROUND TRUTH", "PLACEBO + REPLICATION", "ROBUSTNESS"]
    dtexts = [
        "Exact linear score, R-squared about 1.000 in all three editions. Exact Shapley and a correctly "
        "specified linear model recover every weight perfectly.",
        "The no-reform 2024 to 2025 transition shows about 0% methodology, versus 79%. The 2025 edition "
        "reproduces the same over-attribution (Academic Reputation about 0.64).",
        "The bias is essentially seed-invariant across 20 random seeds and appears in two model families: "
        "random forest and gradient boosting.",
    ]
    for px, t, tx, c in zip(dpos, dtitles, dtexts, [TEAL, GOLD, CORAL]):
        axd = panel([px, 0.205, dw, 0.150], fc="white", ec=c)
        axd.add_patch(plt.Rectangle((0, 0.84), 1, 0.16, transform=axd.transAxes, fc=c, ec=c))
        axd.text(0.5, 0.92, t, transform=axd.transAxes, ha="center", va="center",
                 fontsize=8.3, fontweight="bold", color="white")
        axd.text(0.07, 0.76, textwrap.fill(tx, width=33), transform=axd.transAxes,
                 ha="left", va="top", fontsize=7.7)

    # --- Takeaway ---
    axT = panel([0.07, 0.088, 0.86, 0.088], fc=NAVY, ec=NAVY)
    axT.text(0.5, 0.5, textwrap.fill(
        "TAKEAWAY:  a high-accuracy model paired with a popular explainer can be confidently wrong about "
        "what a reform rewarded when indicators are correlated. Prefer correctly specified models, and "
        "validate explanations against ground truth before trusting them in institutional analytics.",
        width=96), transform=axT.transAxes, ha="center", va="center", fontsize=9, color="white")

    fig.text(0.5, 0.055,
             "Data: QS World University Rankings, 2023 to 2025.    Reproducible code: github.com/maxqcorp/qsrankingproject",
             ha="center", va="center", fontsize=7.5, color="#666666")

    fig.savefig(f"{OUT}/graphical_abstract_a4.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
