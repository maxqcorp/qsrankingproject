"""
Shared data layer for the QS faithful-XAI decomposition study.

Loads the 2023/2024/2025 QS workbooks into a tidy panel, defines the QS
weighting schemes (old vs new methodology), reconstructs the published
overall score from indicator scores, and provides the exact analytic
decomposition of 2023->2024 score change into methodology vs performance.

All downstream scripts import from here so the data contract is single-source.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "Dataset")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

FILES = {
    2023: "2023 QS World University Rankings V2.1 (For qs.com).xlsx",
    2024: "2024 QS World University Rankings 1.2 (For qs.com).xlsx",
    2025: "2025 QS World University Rankings 2.2 (For qs.com)(1).xlsx",
}

# short key -> source score column
IND_COL = {
    "ar": "ar score", "er": "er score", "fsr": "fsr score", "cpf": "cpf score",
    "ifr": "ifr score", "isr": "isr score", "irn": "irn score", "ger": "ger score",
    "sus": "SUS SCORE",
}
IND = list(IND_COL)  # canonical order
IND_LABEL = {
    "ar": "Academic Reputation", "er": "Employer Reputation",
    "fsr": "Faculty-Student", "cpf": "Citations/Faculty",
    "ifr": "Intl Faculty", "isr": "Intl Students",
    "irn": "Intl Research Network", "ger": "Employment Outcomes",
    "sus": "Sustainability",
}
OVERALL_COL = {2023: "score scaled", 2024: "Overall Score", 2025: "Overall Score"}

# QS published weights (fractions). IRN & GER existed in 2023 data but were
# unweighted; SUS did not exist before 2024.
W_OLD = {"ar": .40, "er": .10, "fsr": .20, "cpf": .20, "ifr": .05, "isr": .05,
         "irn": .00, "ger": .00, "sus": .00}
W_NEW = {"ar": .30, "er": .15, "fsr": .10, "cpf": .20, "ifr": .05, "isr": .05,
         "irn": .05, "ger": .05, "sus": .05}
COMMON = ["ar", "er", "fsr", "cpf", "ifr", "isr", "irn", "ger"]  # present both years


def parse_rank(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("=", "").replace(",", "")
    if s in ("", "nan"):
        return np.nan
    if "-" in s:
        a, b = s.split("-")[:2]
        try:
            return (float(a) + float(b)) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return np.nan


def load_year(year):
    path = os.path.join(DATA, FILES[year])
    raw = pd.read_excel(path, header=None)
    hr = next(i for i in range(8)
              if "institution" in [str(x).strip().lower() for x in raw.iloc[i]])
    df = pd.read_excel(path, header=hr)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["institution"].notna()].copy()
    df["institution"] = df["institution"].astype(str).str.strip()

    loccode = df["location code"].astype(str).str.strip()
    if loccode.str.len().median() <= 3:          # 2-letter codes -> country in 'location'
        df["country"] = df["location"].astype(str).str.strip()
        df["region"] = np.nan
    else:                                         # 2025 layout: country in 'location code'
        df["country"] = loccode
        df["region"] = df["location"].astype(str).str.strip()

    out = pd.DataFrame({"institution": df["institution"],
                        "country": df["country"], "region": df["region"],
                        "research": df.get("research"), "size": df.get("size"),
                        "rank_num": df["rank display"].apply(parse_rank)})
    for k, col in IND_COL.items():
        out[k] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    ov = OVERALL_COL[year]
    out["overall_pub"] = pd.to_numeric(df[ov], errors="coerce") if ov in df.columns else np.nan
    return out


def recover_weights(year):
    """Recover the QS weighting scheme for one edition by no-intercept OLS of the
    published overall score on indicator scores. Indicators absent that edition
    (e.g. Sustainability in 2023) are excluded from the fit and reported as 0.
    Returns (weights_dict over IND, R^2, n)."""
    f = load_year(year)
    cols = [k for k in IND if f[k].notna().any()]
    m = f.dropna(subset=cols + ["overall_pub"])
    X = m[cols].values.astype(float)
    y = m["overall_pub"].values.astype(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    w = {k: 0.0 for k in IND}
    for k, c in zip(cols, coef):
        w[k] = float(c)
    pred = X @ coef
    r2 = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return w, r2, len(m)


def decompose_2024_2025(p):
    """Placebo decomposition: 2024->2025 is a SAME-methodology transition (both
    editions use W_NEW, no indicator added), so the methodology component is
    zero by construction and all movement is performance. Used as a no-reform
    control for the 2023->2024 result. Identity is exact."""
    x24 = {k: p[f"{k}_2024"].values for k in IND}
    x25 = {k: p[f"{k}_2025"].values for k in IND}

    s_24 = sum(W_NEW[k] * x24[k] for k in IND)
    s_25 = sum(W_NEW[k] * x25[k] for k in IND)

    methodology = sum((W_NEW[k] - W_NEW[k]) * x24[k] for k in IND)   # identically 0
    performance = sum(W_NEW[k] * (x25[k] - x24[k]) for k in IND)
    dS = s_25 - s_24

    out = p[["institution", "country", "region"]].copy()
    out["score_2024"] = s_24
    out["score_2025"] = s_25
    out["dS_total"] = dS
    out["methodology"] = methodology
    out["performance"] = performance
    out["identity_err"] = dS - (methodology + performance)
    return out


def weighted_score(df, weights, keys=IND):
    s = np.zeros(len(df))
    for k in keys:
        if weights[k] == 0:      # skip dormant indicators (avoids 0*NaN when absent)
            continue
        s = s + weights[k] * df[k].values
    return s


def build_panel():
    """Wide panel of institutions matched across all three years, with
    per-year indicator scores and published overall scores."""
    frames = {y: load_year(y) for y in FILES}
    p = frames[2023].add_suffix("_2023").rename(columns={"institution_2023": "institution"})
    for y in (2024, 2025):
        f = frames[y].add_suffix(f"_{y}").rename(columns={f"institution_{y}": "institution"})
        p = p.merge(f, on="institution", how="inner")
    p["country"] = p["country_2025"].fillna(p["country_2023"])
    p["region"] = p["region_2025"]
    p["research"] = p["research_2024"].fillna(p["research_2023"])
    p["size"] = p["size_2024"].fillna(p["size_2023"])
    return p


def decompose_2023_2024(p):
    """Exact decomposition of each institution's 2023->2024 change in the QS
    overall score into a methodology component (reweighting of retained
    indicators + activation of the new Sustainability dimension) and a
    performance component (movement on retained indicators under new weights).

    Identity (exact):  dS = methodology + performance
    """
    x23 = {k: p[f"{k}_2023"].values for k in IND}
    x24 = {k: p[f"{k}_2024"].values for k in IND}

    s_old_23 = sum(W_OLD[k] * x23[k] for k in COMMON)              # 2023 score, old rules
    s_new_24 = sum(W_NEW[k] * x24[k] for k in COMMON) + W_NEW["sus"] * x24["sus"]

    reweight = sum((W_NEW[k] - W_OLD[k]) * x23[k] for k in COMMON)  # weights change, perf fixed
    activate_sus = W_NEW["sus"] * x24["sus"]                        # new dimension switched on
    performance = sum(W_NEW[k] * (x24[k] - x23[k]) for k in COMMON) # perf moves, new weights

    methodology = reweight + activate_sus
    dS = s_new_24 - s_old_23

    out = p[["institution", "country", "region"]].copy()
    out["score_old_2023"] = s_old_23
    out["score_new_2024"] = s_new_24
    out["dS_total"] = dS
    out["methodology"] = methodology
    out["reweight"] = reweight
    out["activate_sus"] = activate_sus
    out["performance"] = performance
    out["identity_err"] = dS - (methodology + performance)
    return out


if __name__ == "__main__":
    # ---- validation ----
    for y in FILES:
        f = load_year(y)
        w = W_OLD if y == 2023 else W_NEW
        m = f.dropna(subset=[k for k in IND if w[k] > 0] + ["overall_pub"])
        recon = weighted_score(m, w)
        err = np.abs(recon - m["overall_pub"].values)
        print(f"{y}: canonical-weight reconstruction vs published overall "
              f"(n={len(m)}): max|err|={err.max():.3f}, mean|err|={err.mean():.4f}")

    p = build_panel()
    print(f"\nMatched panel: {len(p)} institutions")
    have = p.dropna(subset=[f"{k}_2023" for k in COMMON] +
                          [f"{k}_2024" for k in COMMON] + ["sus_2024"])
    print(f"Complete common-indicator + 2024 sustainability data: {len(have)}")

    dec = decompose_2023_2024(have)
    print(f"\nDecomposition identity max|err|: {np.abs(dec['identity_err']).max():.2e}")
    tot = dec["methodology"].abs() + dec["performance"].abs()
    print(f"Mean |methodology| score pts: {dec['methodology'].abs().mean():.2f}  "
          f"(reweight {dec['reweight'].abs().mean():.2f}, sus {dec['activate_sus'].abs().mean():.2f})")
    print(f"Mean |performance| score pts: {dec['performance'].abs().mean():.2f}")
    share = dec["methodology"].abs().sum() / tot.sum()
    print(f"Share of total absolute movement attributable to methodology: {share:.1%}")
    print(f"Variance of dS: {dec['dS_total'].var():.2f} | "
          f"var(methodology)={dec['methodology'].var():.2f} | "
          f"var(performance)={dec['performance'].var():.2f}")
