#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="${1:-}"

if [[ -z "$REMOTE_URL" ]]; then
  echo "Usage: bash scripts/github_upload_commands.sh git@github.com:YOUR_USERNAME/binderbench.git"
  exit 1
fi

git init
git add .
git commit -m "Initial reproducible hybrid PPI binder benchmark"
git branch -M main
git remote add origin "$REMOTE_URL"
git push -u origin main
