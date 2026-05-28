#!/usr/bin/env bash
# Build pipeline for the QS explainable-ML study.
#   regenerate analysis + figures  ->  refresh figure bundle  ->  rebuild docx
#
# Usage:
#   ./build.sh              Full build: regenerate figures (needs a venv + Dataset/),
#                           refresh paper/figures/, rebuild manuscript + cover letters.
#   ./build.sh --docx-only  Skip figure regeneration; just refresh the bundle from the
#                           existing figures and rebuild the docx. Fast; no data needed.
#                           Use this after filling in author/funding placeholders.
#
# Python is chosen as: $PYTHON, then ./.venv, then ~/.venvs/qs_ranking, then python3.
set -euo pipefail
cd "$(dirname "$0")"   # always run from the repo root

DOCX_ONLY=0
[ "${1:-}" = "--docx-only" ] && DOCX_ONLY=1

pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for p in ./.venv/bin/python "$HOME/.venvs/qs_ranking/bin/python" python3 python; do
    if [ -x "$p" ] || command -v "$p" >/dev/null 2>&1; then echo "$p"; return; fi
  done
  echo ""
}

OUT=analysis/output
FIGDIR=paper/figures

if [ "$DOCX_ONLY" -eq 0 ]; then
  PY="$(pick_python)"
  if [ -z "$PY" ]; then
    echo "ERROR: no Python interpreter found. Set PYTHON=... or create .venv (see README)." >&2
    exit 1
  fi
  echo "[1/4] Regenerating analysis + figures with: $PY"
  "$PY" analysis/xai_faithfulness.py
  "$PY" analysis/xai_substantive.py
else
  echo "[1/4] --docx-only: skipping figure regeneration"
fi

echo "[2/4] Refreshing figure bundle -> $FIGDIR"
mkdir -p "$FIGDIR"
cp "$OUT/paper_fig1_data_overview.png"      "$FIGDIR/Figure1.png"
cp "$OUT/paper_fig2_decomposition.png"      "$FIGDIR/Figure2.png"
cp "$OUT/paper_fig3_rank_by_region.png"     "$FIGDIR/Figure3.png"
cp "$OUT/paper_fig4_research_intensity.png" "$FIGDIR/Figure4.png"
cp "$OUT/paper_fig5_faithfulness.png"       "$FIGDIR/Figure5.png"
cp "$OUT/paper_fig6_simulator.png"          "$FIGDIR/Figure6.png"
cp "$OUT/paper_fig7_weights_by_year.png"    "$FIGDIR/Figure7.png"
cp "$OUT/paper_fig8_faithfulness_2025.png"  "$FIGDIR/Figure8.png"
cp "$OUT/paper_fig9_robustness.png"         "$FIGDIR/Figure9.png"
cp "$OUT/graphical_abstract.png"            "$FIGDIR/GraphicalAbstract.png"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc not found (needed to rebuild the .docx)." >&2
  exit 1
fi
echo "[3/4] Rebuilding paper/manuscript.docx"
pandoc paper/manuscript_docx.md --citeproc \
  --bibliography paper/references.bib --csl paper/ieee.csl \
  -o paper/manuscript.docx

echo "[4/4] Rebuilding cover letters"
pandoc paper/cover_letter.md    -o paper/cover_letter.docx
pandoc paper/cover_letter_v2.md -o paper/cover_letter_v2.docx

echo "Done. -> paper/manuscript.docx, paper/cover_letter.docx, paper/cover_letter_v2.docx, $FIGDIR/Figure1..9.png + GraphicalAbstract.png"
