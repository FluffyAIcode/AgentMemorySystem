#!/usr/bin/env bash
# Build an arXiv-compliant tarball from reports/paper/.
#
# Usage (run from the KakeyaLattice repo root):
#   bash dissemination/kakeyalattice/arxiv/build_tarball.sh
# Produces: dissemination/kakeyalattice/arxiv/arxiv_submission.tar.gz
#
# The tarball contains:
#   - kakeyalattice.tex          (main source)
#   - any .bbl / .bib / figures / style files from reports/paper/
# and omits build artefacts listed in reports/paper/.gitignore.
#
# Requirements: bash, tar, grep, awk; pdflatex+bibtex only if you want
# to pre-build the .bbl (recommended, arXiv builds faster with .bbl included).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"   # dissemination/kakeyalattice/arxiv/ -> repo root
PAPER_DIR="$REPO_ROOT/reports/paper"
OUT="$HERE/arxiv_submission.tar.gz"
STAGE="$(mktemp -d)"

if [[ ! -f "$PAPER_DIR/kakeyalattice.tex" ]]; then
    echo "ERROR: expected $PAPER_DIR/kakeyalattice.tex" >&2
    echo "Run this script from inside the KakeyaLattice repo (FluffyAIcode/LLM-KV--Cache-compress)." >&2
    exit 1
fi

echo "==> Staging paper sources in $STAGE"
cp "$PAPER_DIR"/*.tex "$STAGE/"
cp "$PAPER_DIR"/*.bib "$STAGE/" 2>/dev/null || true
cp "$PAPER_DIR"/*.cls "$STAGE/" 2>/dev/null || true
cp "$PAPER_DIR"/*.sty "$STAGE/" 2>/dev/null || true

# Figures subdirs (common layouts)
for d in figures figs images img; do
    if [[ -d "$PAPER_DIR/$d" ]]; then
        cp -r "$PAPER_DIR/$d" "$STAGE/"
    fi
done

# Try to pre-build the .bbl so arXiv's build path is shorter.
if command -v pdflatex >/dev/null && command -v bibtex >/dev/null; then
    echo "==> Pre-building .bbl with pdflatex+bibtex"
    pushd "$STAGE" >/dev/null
    pdflatex -interaction=nonstopmode kakeyalattice.tex >/dev/null || true
    bibtex kakeyalattice >/dev/null || true
    pdflatex -interaction=nonstopmode kakeyalattice.tex >/dev/null || true
    pdflatex -interaction=nonstopmode kakeyalattice.tex >/dev/null || true
    # Remove intermediate artefacts; keep .bbl
    rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.blg
    popd >/dev/null
else
    echo "WARN: pdflatex/bibtex not found — arXiv will build the .bbl server-side."
fi

echo "==> Creating tarball $OUT"
rm -f "$OUT"
tar -czf "$OUT" -C "$STAGE" .
ls -lh "$OUT"

echo
echo "Next steps:"
echo "  1. Go to https://arxiv.org/submit and start a new submission"
echo "  2. Primary category: cs.LG (see metadata.yaml)"
echo "  3. Upload $OUT as 'tar archive of sources'"
echo "  4. Paste title / abstract / comments from metadata.yaml"
echo "  5. License: CC BY 4.0 (recommended)"
echo
echo "If this is your first cs.LG submission, request endorsement first:"
echo "  see dissemination/kakeyalattice/arxiv/endorsement_request.md"
