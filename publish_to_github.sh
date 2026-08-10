#!/usr/bin/env bash
set -e

# Usage:
#   1) Create an empty GitHub repo in the browser, for example: numpy-tiny-dataset-assistant
#   2) Replace YOUR_USERNAME and YOUR_REPO below or pass the URL as the first argument:
#      ./publish_to_github.sh https://github.com/YOUR_USERNAME/YOUR_REPO.git

REMOTE_URL="${1:-https://github.com/YOUR_USERNAME/YOUR_REPO.git}"

python check_github_ready.py

git init
git add .gitignore README.md requirements.txt prepare_dataset.py check_github_ready.py publish_to_github.sh numpy_tiny_llm_export/run_model.py numpy_tiny_llm_export/chat.py numpy_tiny_llm_export/tokenizer.json numpy_tiny_llm_export/model_weights.npz numpy_tiny_llm_export/assistant_knowledge.json numpy_tiny_llm_export/gpt4all_full_dataset_manifest.json

git commit -m "Add NumPy dataset-backed assistant"
git branch -M main
git remote add origin "$REMOTE_URL"
git push -u origin main
