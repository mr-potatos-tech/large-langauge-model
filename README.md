# NumPy Tiny Dataset Assistant

This project is a tiny educational assistant built with NumPy plus a dataset-backed retrieval layer.

It uses the GPT4All-J prompt-generations dataset from Hugging Face:
https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations/tree/main/data

## Important GitHub note

You can host the code on GitHub directly, but do not commit the full generated dataset index to a normal GitHub repository.

The file `numpy_tiny_llm_export/gpt4all_full_dataset_index.jsonl` is about 1.2GB. GitHub normal repositories have a 100MB per-file limit, so this file is ignored by `.gitignore`.

Recommended workflow:
1. Commit the code and small model files to GitHub.
2. After cloning the repo, run `python prepare_dataset.py` to download the dataset shards and rebuild the full local index.

## Setup

```bash
pip install -r requirements.txt
python prepare_dataset.py
```

`prepare_dataset.py` downloads all parquet shards and builds:

```text
numpy_tiny_llm_export/gpt4all_full_dataset_index.jsonl
numpy_tiny_llm_export/gpt4all_full_dataset_manifest.json
```

## Run one-shot

```bash
python numpy_tiny_llm_export/run_model.py "explain recursion in simple terms"
```

## Run interactive chat

```bash
python numpy_tiny_llm_export/chat.py
```

Then type a question, for example:

```text
how do I make a good pasta sauce
what is python?
write a short poem about the moon
```

## Publish to GitHub

Create an empty GitHub repository, then run:

```bash
./publish_to_github.sh https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

Or manually:

```bash
python check_github_ready.py
git init
git add .
git commit -m "Add NumPy dataset-backed assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Because `.gitignore` excludes the huge dataset/index files, `git add .` should not include them.

## Files

```text
numpy_tiny_llm_export/run_model.py          one-shot CLI
numpy_tiny_llm_export/chat.py               interactive CLI
numpy_tiny_llm_export/model_weights.npz     tiny NumPy RNN weights
numpy_tiny_llm_export/tokenizer.json        tokenizer vocabulary
numpy_tiny_llm_export/assistant_knowledge.json small assistant knowledge
prepare_dataset.py                          downloads/builds full dataset index
requirements.txt                            Python dependencies
```

## Limitations

This is not a real large language model. The tiny NumPy RNN is educational. The useful answers come mainly from searching the GPT4All-J dataset index.

Searching the full JSONL index scans 808,812 rows and can be slow. For faster production use, replace the JSONL scan with a vector database, SQLite FTS5, LanceDB, FAISS, or BM25 index.
