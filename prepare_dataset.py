"""
Prepare local dataset indexes for the dataset-backed NumPy assistant.

Default behavior:
    python prepare_dataset.py

This builds/reuses:
    - GPT4All-J full index
    - UltraChat sample index

Full UltraChat indexing can be large/slow:
    python prepare_dataset.py --include-ultrachat-full

The generated dataset indexes are intentionally ignored by git because they can be large.
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, hf_hub_url

EXPORT_DIR = Path('numpy_tiny_llm_export')
GPT4ALL_REPO_ID = 'nomic-ai/gpt4all-j-prompt-generations'
GPT4ALL_DATASET_DIR = Path('datasets/gpt4all_j_prompt_generations_all')
GPT4ALL_INDEX_PATH = EXPORT_DIR / 'gpt4all_full_dataset_index.jsonl'
GPT4ALL_MANIFEST_PATH = EXPORT_DIR / 'gpt4all_full_dataset_manifest.json'

ULTRACHAT_REPO_ID = 'openbmb/UltraChat'
ULTRACHAT_DATASET_DIR = Path('datasets/ultrachat')
ULTRACHAT_SAMPLE_INDEX_PATH = EXPORT_DIR / 'ultrachat_sample_index.jsonl'
ULTRACHAT_FULL_INDEX_PATH = EXPORT_DIR / 'ultrachat_full_index.jsonl'
ULTRACHAT_MANIFEST_PATH = EXPORT_DIR / 'ultrachat_manifest.json'


def clean_text(text, max_len):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if text.lower() in {'', 'nan', 'none', 'null'}:
        return ''
    return text[:max_len]


def build_gpt4all_index():
    EXPORT_DIR.mkdir(exist_ok=True)
    GPT4ALL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    repo_files = list_repo_files(GPT4ALL_REPO_ID, repo_type='dataset')
    parquet_files = sorted([f for f in repo_files if f.startswith('data/') and f.endswith('.parquet')])
    local_paths = []
    for remote_path in parquet_files:
        local_path = GPT4ALL_DATASET_DIR / Path(remote_path).name
        local_paths.append(local_path)
        if not local_path.exists() or local_path.stat().st_size == 0:
            url = hf_hub_url(repo_id=GPT4ALL_REPO_ID, filename=remote_path, repo_type='dataset')
            print(f'Downloading {remote_path} ...')
            urllib.request.urlretrieve(url, local_path)
    expected_files_info = [{'file': str(p), 'size': p.stat().st_size} for p in local_paths]
    if GPT4ALL_INDEX_PATH.exists() and GPT4ALL_MANIFEST_PATH.exists():
        try:
            old = json.loads(GPT4ALL_MANIFEST_PATH.read_text(encoding='utf-8'))
            if old.get('files') == expected_files_info and old.get('index_rows', 0) > 0:
                print('GPT4All-J index already exists.')
                return
        except Exception:
            pass
    total_rows_seen = 0
    indexed_rows = 0
    source_counts = {}
    columns_seen = set()
    with GPT4ALL_INDEX_PATH.open('w', encoding='utf-8') as out:
        for shard_i, path in enumerate(local_paths, start=1):
            pf = pq.ParquetFile(path)
            print(f'Processing GPT4All shard {shard_i}/{len(local_paths)}: {path.name}')
            for rg in range(pf.num_row_groups):
                df = pf.read_row_group(rg).to_pandas()
                columns_seen.update(df.columns)
                for row in df.itertuples(index=False):
                    row_dict = row._asdict()
                    total_rows_seen += 1
                    prompt = clean_text(row_dict.get('prompt', ''), 900)
                    response = clean_text(row_dict.get('response', ''), 1400)
                    source = clean_text(row_dict.get('source', ''), 120)
                    if not prompt or not response:
                        continue
                    source_counts[source] = source_counts.get(source, 0) + 1
                    out.write(json.dumps({'id': indexed_rows, 'dataset': GPT4ALL_REPO_ID, 'shard': path.name, 'row_group': rg, 'prompt': prompt, 'response': response, 'source': source}, ensure_ascii=False) + '\n')
                    indexed_rows += 1
                if (rg + 1) % 25 == 0 or rg == pf.num_row_groups - 1:
                    print(f'  row group {rg + 1}/{pf.num_row_groups} | indexed {indexed_rows:,}')
    manifest = {
        'repo_id': GPT4ALL_REPO_ID,
        'dataset_tree_url': 'https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations/tree/main/data',
        'files': expected_files_info,
        'parquet_files': parquet_files,
        'columns_seen': sorted(columns_seen),
        'total_rows_seen': total_rows_seen,
        'index_rows': indexed_rows,
        'source_counts_top': sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:30],
        'index_path': str(GPT4ALL_INDEX_PATH),
    }
    GPT4ALL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'GPT4All-J indexed rows: {indexed_rows:,}')


def ultrachat_pairs_from_record(obj):
    data = obj.get('data', [])
    if not isinstance(data, list):
        return []
    pairs = []
    for i in range(0, len(data) - 1, 2):
        prompt = clean_text(data[i], 900)
        response = clean_text(data[i + 1], 1400)
        if prompt and response:
            pairs.append((prompt, response))
    return pairs


def build_ultrachat_index(full=False, sample_dialogues=2500):
    EXPORT_DIR.mkdir(exist_ok=True)
    ULTRACHAT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in list_repo_files(ULTRACHAT_REPO_ID, repo_type='dataset') if f.startswith('train_') and f.endswith('.jsonl')])
    index_path = ULTRACHAT_FULL_INDEX_PATH if full else ULTRACHAT_SAMPLE_INDEX_PATH
    if index_path.exists() and index_path.stat().st_size > 0:
        print(f'UltraChat index already exists: {index_path}')
        return
    indexed_pairs = 0
    dialogues_seen = 0
    files_used = []
    with index_path.open('w', encoding='utf-8') as out:
        for filename in files:
            files_used.append(filename)
            url = hf_hub_url(repo_id=ULTRACHAT_REPO_ID, filename=filename, repo_type='dataset')
            print(f'Streaming UltraChat {filename} ...')
            with urllib.request.urlopen(url, timeout=180) as resp:
                for raw in resp:
                    if (not full) and dialogues_seen >= sample_dialogues:
                        break
                    try:
                        obj = json.loads(raw.decode('utf-8'))
                    except Exception:
                        continue
                    dialogues_seen += 1
                    for prompt, response in ultrachat_pairs_from_record(obj):
                        out.write(json.dumps({'id': indexed_pairs, 'dataset': ULTRACHAT_REPO_ID, 'file': filename, 'dialogue_id': obj.get('id', ''), 'prompt': prompt, 'response': response, 'source': ULTRACHAT_REPO_ID}, ensure_ascii=False) + '\n')
                        indexed_pairs += 1
                    if dialogues_seen % 10000 == 0:
                        print(f'  dialogues {dialogues_seen:,} | pairs {indexed_pairs:,}')
            if (not full) and dialogues_seen >= sample_dialogues:
                break
    manifest = {
        'repo_id': ULTRACHAT_REPO_ID,
        'dataset_url': 'https://huggingface.co/datasets/openbmb/UltraChat',
        'files': files,
        'files_used': files_used,
        'index_path': str(index_path),
        'full': full,
        'dialogues_seen': dialogues_seen,
        'pairs_indexed': indexed_pairs,
    }
    ULTRACHAT_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'UltraChat indexed pairs: {indexed_pairs:,}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-gpt4all', action='store_true')
    parser.add_argument('--skip-ultrachat', action='store_true')
    parser.add_argument('--include-ultrachat-full', action='store_true', help='Build full UltraChat index instead of sample; can be large/slow.')
    parser.add_argument('--ultrachat-sample-dialogues', type=int, default=2500)
    args = parser.parse_args()
    if not args.skip_gpt4all:
        build_gpt4all_index()
    if not args.skip_ultrachat:
        build_ultrachat_index(full=args.include_ultrachat_full, sample_dialogues=args.ultrachat_sample_dialogues)


if __name__ == '__main__':
    main()
