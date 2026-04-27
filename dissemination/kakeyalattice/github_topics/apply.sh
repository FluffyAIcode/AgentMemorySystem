#!/usr/bin/env bash
# Apply GitHub topics + description to FluffyAIcode/LLM-KV--Cache-compress.
# Requires: gh CLI authenticated as repo owner (or someone with admin rights).
# Idempotent — safe to re-run.

set -euo pipefail

REPO="${KAKEYA_REPO:-FluffyAIcode/LLM-KV--Cache-compress}"
HOMEPAGE="${KAKEYA_HOMEPAGE:-https://github.com/FluffyAIcode/LLM-KV--Cache-compress/releases/tag/v1.5}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESCRIPTION="$(cat "$HERE/description.txt")"

echo "==> Setting description and homepage on $REPO"
gh api --method PATCH "repos/$REPO" \
    -f description="$DESCRIPTION" \
    -f homepage="$HOMEPAGE" \
    -F has_issues=true \
    -F has_discussions=true \
    >/dev/null

echo "==> Setting topics on $REPO"
# Replace topics wholesale with the curated list from topics.json.
gh api --method PUT "repos/$REPO/topics" \
    -H "Accept: application/vnd.github.mercy-preview+json" \
    --input "$HERE/topics.json" \
    >/dev/null

echo "==> Done. Verify at: https://github.com/$REPO"
gh api "repos/$REPO" --jq '{full_name, description, homepage, topics}'
