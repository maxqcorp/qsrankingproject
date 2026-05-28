"""
QS World University Rankings — Risers & Fallers study (2023 -> 2025).

Builds a cleaned 3-year panel of universities matched across all three years,
computes rank trajectories, identifies the biggest risers and fallers, and
explains the moves via changes in the underlying QS indicators.

Run:  python3 analysis/risers_fallers.py
Outputs land in analysis/output/.
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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

# QS indicator score columns (0-100, higher = better) present across years.
INDICATORS = {
    "ar score":  "Academic Reputation",
    "er score":  "Employer Reputation",
    "fsr score": "Faculty-Student Ratio",
    "cpf score": "Citations per Faculty",
    "ifr score": "International Faculty",
    "isr score": "International Students",
    "irn score": "Intl Research Network",
    "ger score": "Employment Outcomes",
    "SUS SCORE": "Sustainability",     # 2024 & 2025 only
}


def parse_rank(val):
    """Convert a QS rank display ('1', '  3=', '601-610', '1201+') to a numeric rank.
    Banded ranks become the band midpoint; ties/'=' and whitespace are stripped."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("=", "").replace(",", "")
    if s == "" or s.lower() == "nan":
        return np.nan
    if "-" in s:  # band like 601-610
        a, b = s.split("-")[:2]
        try:
            return (float(a) + float(b)) / 2.0
        except ValueError:
            return np.nan
    s = s.replace("+", "")  # tail band like 1201+
    try:
        return float(s)
    except ValueError:
        return np.nan


def find_header_row(raw):
    for i in range(min(8, len(raw))):
        if "institution" in [str(x).strip().lower() for x in raw.iloc[i]]:
            return i
    raise ValueError("header row with 'institution' not found")


def load_year(year):
    path = os.path.join(DATA, FILES[year])
    raw = pd.read_excel(path, sheet_name=0, header=None)
    hr = find_header_row(raw)
    df = pd.read_excel(path, sheet_name=0, header=hr)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["institution"].notna()].copy()
    df["institution"] = df["institution"].astype(str).str.strip()

    # --- location handling: 2023/24 -> code(2-letter)+country; 2025 -> country+region ---
    loccode = df["location code"].astype(str).str.strip()
    if loccode.str.len().median() <= 3:          # genuine 2-letter codes
        df["country"] = df["location"].astype(str).str.strip()
        df["region"] = np.nan
    else:                                         # 2025 layout: 'location code' holds country
        df["country"] = loccode
        df["region"] = df["location"].astype(str).str.strip()

    df["rank_num"] = df["rank display"].apply(parse_rank)

    keep = ["institution", "country", "region", "rank_num"]
    for col in INDICATORS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            keep.append(col)
    out = df[keep].copy()
    out = out.rename(columns={c: f"{c}|{year}" for c in keep if c != "institution"})
    return out


def build_panel():
    frames = {y: load_year(y) for y in FILES}
    panel = frames[2023]
    for y in (2024, 2025):
        panel = panel.merge(frames[y], on="institution", how="inner")
    # carry a single country label (prefer 2025, fall back)
    panel["country"] = panel["country|2025"].fillna(panel["country|2023"])
    panel["region"] = panel["region|2025"]
    return panel


def main():
    panel = build_panel()
    print(f"Matched universities across all 3 years: {len(panel)}")

    # ---- rank trajectories -------------------------------------------------
    r23, r24, r25 = panel["rank_num|2023"], panel["rank_num|2024"], panel["rank_num|2025"]
    # positive = improved (moved toward #1)
    panel["change_3yr"] = r23 - r25
    panel["change_24_25"] = r24 - r25

    panel_out = panel.copy()
    panel_out.to_csv(os.path.join(OUT, "qs_panel_2023_2025.csv"), index=False)

    valid3 = panel[panel["rank_num|2023"].notna() & panel["rank_num|2025"].notna()].copy()
    print(f"Universities ranked in both 2023 and 2025: {len(valid3)}")

    risers = valid3.sort_values("change_3yr", ascending=False).head(15)
    fallers = valid3.sort_values("change_3yr", ascending=True).head(15)

    cols_show = ["institution", "country", "rank_num|2023", "rank_num|2024",
                 "rank_num|2025", "change_3yr"]
    risers[cols_show].to_csv(os.path.join(OUT, "top_risers_3yr.csv"), index=False)
    fallers[cols_show].to_csv(os.path.join(OUT, "top_fallers_3yr.csv"), index=False)

    # ---- indicator deltas for movers --------------------------------------
    def indicator_deltas(row):
        out = {}
        for col, label in INDICATORS.items():
            a, b = f"{col}|2023", f"{col}|2025"
            if a in panel.columns and b in panel.columns:
                va, vb = row.get(a), row.get(b)
                if pd.notna(va) and pd.notna(vb):
                    out[label] = vb - va
        return out

    # ---- console summary ---------------------------------------------------
    def fmt(df, sign):
        lines = []
        for _, r in df.iterrows():
            d = indicator_deltas(r)
            d = {k: v for k, v in d.items() if (v > 0) == (sign > 0)}
            top = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            drivers = ", ".join(f"{k} {v:+.1f}" for k, v in top) or "—"
            lines.append(f"  {r['institution'][:42]:42} {r['country'][:14]:14} "
                         f"{int(r['rank_num|2023']):>5} -> {int(r['rank_num|2025']):>5} "
                         f"({r['change_3yr']:+.0f})  | {drivers}")
        return "\n".join(lines)

    report = []
    report.append("=" * 100)
    report.append("QS RANKINGS — RISERS & FALLERS, 2023 -> 2025")
    report.append("=" * 100)
    report.append(f"Panel: {len(panel)} universities matched across all 3 years; "
                  f"{len(valid3)} have a numeric rank in both 2023 and 2025.")
    report.append("Note: ranks beyond ~400 are banded (e.g. 601-610); midpoints used. "
                  "Positive change = rose toward #1.")
    report.append("")
    report.append("TOP 15 RISERS (places gained, with top favourable indicator changes)")
    report.append("-" * 100)
    report.append(fmt(risers, +1))
    report.append("")
    report.append("TOP 15 FALLERS (places lost, with top unfavourable indicator changes)")
    report.append("-" * 100)
    report.append(fmt(fallers, -1))
    report.append("")
    # country-level net movement context
    cc = (valid3.groupby("country")["change_3yr"]
          .agg(["mean", "count"]).query("count >= 5")
          .sort_values("mean", ascending=False))
    report.append("COUNTRY NET MOVEMENT (avg places gained, countries with >=5 ranked unis)")
    report.append("-" * 100)
    report.append("  Biggest gainers: " +
                  ", ".join(f"{i} ({v:+.0f})" for i, v in cc["mean"].head(6).items()))
    report.append("  Biggest losers : " +
                  ", ".join(f"{i} ({v:+.0f})" for i, v in cc["mean"].tail(6).items()))
    text = "\n".join(report)
    print(text)
    with open(os.path.join(OUT, "findings.txt"), "w") as f:
        f.write(text + "\n")

    # ---- charts ------------------------------------------------------------
    make_charts(panel, valid3, risers, fallers, indicator_deltas)
    print(f"\nArtifacts written to: {OUT}")


def make_charts(panel, valid3, risers, fallers, indicator_deltas):
    plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

    # 1) Top risers & fallers bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, df, color, title in [
        (axes[0], risers.iloc[::-1], "#2a9d8f", "Top 15 Risers (places gained, 2023→2025)"),
        (axes[1], fallers.iloc[::-1], "#e76f51", "Top 15 Fallers (places lost, 2023→2025)"),
    ]:
        labels = [f"{n[:30]}" for n in df["institution"]]
        ax.barh(labels, df["change_3yr"], color=color)
        ax.set_title(title)
        ax.set_xlabel("Change in rank position (+ = improved)")
        ax.axvline(0, color="#444", lw=0.8)
        for y, v in enumerate(df["change_3yr"]):
            ax.text(v, y, f" {v:+.0f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_risers_fallers_bars.png"), bbox_inches="tight")
    plt.close(fig)

    # 2) Trajectory (bump) chart for notable movers with reasonably precise ranks
    precise = valid3[valid3["rank_num|2025"] <= 420].copy()
    pick = pd.concat([
        precise.sort_values("change_3yr", ascending=False).head(5),
        precise.sort_values("change_3yr", ascending=True).head(5),
    ])
    fig, ax = plt.subplots(figsize=(12, 7))
    years = [2023, 2024, 2025]
    for _, r in pick.iterrows():
        ys = [r["rank_num|2023"], r["rank_num|2024"], r["rank_num|2025"]]
        rising = r["change_3yr"] >= 0
        ax.plot(years, ys, marker="o",
                color="#2a9d8f" if rising else "#e76f51", alpha=0.85)

    # de-clutter right-side labels: enforce a minimum vertical gap, draw connectors
    pick_sorted = pick.sort_values("rank_num|2025")
    span = pick["rank_num|2025"].max() - pick["rank_num|2025"].min()
    min_gap = max(span * 0.06, 12)
    label_y, last = [], -1e9
    for y in pick_sorted["rank_num|2025"]:
        y = max(y, last + min_gap)
        label_y.append(y)
        last = y
    for (_, r), ly in zip(pick_sorted.iterrows(), label_y):
        rising = r["change_3yr"] >= 0
        c = "#1d6f64" if rising else "#b5432a"
        ax.plot([2025, 2025.04], [r["rank_num|2025"], ly], color=c, lw=0.6, alpha=0.6)
        ax.text(2025.06, ly, f" {r['institution'][:30]}",
                va="center", fontsize=8, color=c)

    ax.invert_yaxis()
    ax.set_xticks(years)
    ax.set_title("Rank trajectories of notable movers (top-420, lower = better)")
    ax.set_ylabel("World rank")
    ax.set_xlim(2022.8, 2026.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "02_trajectories.png"), bbox_inches="tight")
    plt.close(fig)

    # 3) Indicator-delta heatmap for top 10 risers + top 10 fallers
    movers = pd.concat([risers.head(10), fallers.head(10)])
    labels = [l for c, l in INDICATORS.items() if f"{c}|2023" in panel.columns]
    mat, names = [], []
    for _, r in movers.iterrows():
        d = indicator_deltas(r)
        mat.append([d.get(l, np.nan) for l in labels])
        tag = "▲" if r["change_3yr"] >= 0 else "▼"
        names.append(f"{tag} {r['institution'][:28]}")
    mat = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 9))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Indicator score change 2023→2025 (green = improved) — top movers")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Δ score")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "03_indicator_deltas.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
