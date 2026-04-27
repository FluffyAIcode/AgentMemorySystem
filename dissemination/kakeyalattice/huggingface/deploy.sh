#!/usr/bin/env bash
# Deploy the KakeyaLattice demo to a HuggingFace Space.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login      # needs a write-scope token
#
# Env vars:
#   HF_USER    — your HF username or org (default: FluffyAIcode)
#   HF_SPACE   — Space name (default: kakeyalattice)
#
# Run from the KakeyaLattice repo root.

set -euo pipefail

HF_USER="${HF_USER:-FluffyAIcode}"
HF_SPACE="${HF_SPACE:-kakeyalattice}"
HF_URL="https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_SRC="$HERE/space"

if ! command -v huggingface-cli >/dev/null; then
    echo "Installing huggingface_hub"
    pip install --quiet huggingface_hub
fi

# Verify login.
if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "ERROR: huggingface-cli not authenticated. Run:" >&2
    echo "  huggingface-cli login" >&2
    exit 1
fi

echo "==> Creating Space $HF_URL (idempotent)"
huggingface-cli repo create "$HF_SPACE" --type space --space_sdk gradio \
    --organization "$HF_USER" -y 2>/dev/null || true

TMP="$(mktemp -d)"
echo "==> Cloning Space into $TMP"
git clone "$HF_URL" "$TMP/$HF_SPACE"

echo "==> Copying app.py / requirements.txt / README.md"
cp -v "$SPACE_SRC/app.py" "$TMP/$HF_SPACE/"
cp -v "$SPACE_SRC/requirements.txt" "$TMP/$HF_SPACE/"
cp -v "$SPACE_SRC/README.md" "$TMP/$HF_SPACE/"

cd "$TMP/$HF_SPACE"
git add -A
git -c user.email="dissemination@kakeyalattice.local" \
    -c user.name="KakeyaLattice Dissemination Bot" \
    commit -m "Initial KakeyaLattice codec demo (auto-generated)" || true
git push

echo
echo "==> Space deployed. Live URL:"
echo "    $HF_URL"
echo
echo "First build takes 3-5 minutes. Check status at:"
echo "    $HF_URL/logs"
