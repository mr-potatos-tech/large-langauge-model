#!/usr/bin/env bash
set -euo pipefail

# Secure push helper.
# Do NOT paste your token into this file.
# Usage:
#   export GITHUB_TOKEN='YOUR_NEW_TOKEN_HERE'
#   ./push_to_github_with_token.sh

BRANCH="Large-Language-Model"
REPO_PATH="Mohamedboukerche22/simple-llm.git"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "Error: GITHUB_TOKEN is not set."
  echo "Run: export GITHUB_TOKEN='YOUR_NEW_TOKEN_HERE'"
  exit 1
fi

python check_github_ready.py

git remote set-url origin "https://github.com/${REPO_PATH}" 2>/dev/null || git remote add origin "https://github.com/${REPO_PATH}"

# Use the token only for this one push command. It is not saved in git remote config.
git push -u "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_PATH}" "${BRANCH}"
