#!/bin/bash
set -euo pipefail

# Secure push helper.
# Do NOT paste your token into this file.
# In notebooks, run with Python/getpass below, or in a terminal:
#   export GITHUB_TOKEN='YOUR_NEW_TOKEN_HERE'
#   bash push_to_github_with_token.sh

BRANCH="Large-Language-Model"
REPO_PATH="Mohamedboukerche22/simple-llm.git"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "Error: GITHUB_TOKEN is not set."
  echo "In a terminal: export GITHUB_TOKEN='YOUR_NEW_TOKEN_HERE'"
  echo "In Jupyter: use the getpass Python cell shown in the notebook answer."
  exit 1
fi

python check_github_ready.py

git remote set-url origin "https://github.com/${REPO_PATH}" 2>/dev/null || git remote add origin "https://github.com/${REPO_PATH}"

# Use the token only for this one push command. It is not saved in git remote config.
git push -u "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_PATH}" "${BRANCH}"

# Reset remote to a clean URL without token.
git remote set-url origin "https://github.com/${REPO_PATH}"
