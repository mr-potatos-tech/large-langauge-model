
"""
Prepare the full GPT4All-J dataset index locally.

Why this exists:
GitHub normal repositories cannot store the generated 1.2GB JSONL index directly.
Commit the code to GitHub, then run this script after cloning to rebuild the index.

Usage:
    pip install -r requirements.txt
    python prepare_dataset.py
"""

import json
import re
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, hf_hub_url

REPO_ID = 'nomic-ai/gpt4all-j-prompt-generations'
DATASET_TREE_URL = 'https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations/tree/main/data'
DATASET_DIR = Path('datasets/gpt4all_j_prompt_generations_all')
EXPORT_DIR = Path('numpy_tiny_llm_export')
INDEX_PATH = EXPORT_DIR / 'gpt4all_full_dataset_index.jsonl'
MANIFEST_PATH = EXPORT_DIR / 'gpt4all_full_dataset_manifest.json'


def clean_text(text, max_len):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if text.lower() in {'', 'nan', 'none', 'null'}:
        return ''
    return text[:max_len]


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(exist_ok=True)

    repo_files = list_repo_files(REPO_ID, repo_type='dataset')
    parquet_files = sorted([f for f in repo_files if f.startswith('data/') and f.endswith('.parquet')])
    if not parquet_files:
        raise RuntimeError('No parquet files found in dataset data/ folder.')

    print(f'Found {len(parquet_files)} parquet shards.')
    local_paths = []
    for remote_path in parquet_files:
        local_path = DATASET_DIR / Path(remote_path).name
        local_paths.append(local_path)
        if local_path.exists() and local_path.stat().st_size > 0:
            print(f'Already downloaded: {local_path.name} ({local_path.stat().st_size / (1024**2):.2f} MB)')
            continue
        url = hf_hub_url(repo_id=REPO_ID, filename=remote_path, repo_type='dataset')
        print(f'Downloading {remote_path} ...')
        urllib.request.urlretrieve(url, local_path)
        print(f'Downloaded: {local_path.name} ({local_path.stat().st_size / (1024**2):.2f} MB)')

    expected_files_info = [{'file': str(p), 'size': p.stat().st_size} for p in local_paths]
    if INDEX_PATH.exists() and MANIFEST_PATH.exists():
        try:
            old_manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
            if old_manifest.get('files') == expected_files_info and old_manifest.get('index_rows', 0) > 0:
                print('Index already exists and matches downloaded shards.')
                return
        except Exception:
            pass

    print('Building full-dataset JSONL index. This can take a while...')
    total_rows_seen = 0
    indexed_rows = 0
    source_counts = {}
    columns_seen = set()

    with INDEX_PATH.open('w', encoding='utf-8') as out:
        for shard_i, path in enumerate(local_paths, start=1):
            pf = pq.ParquetFile(path)
            print(f'Processing shard {shard_i}/{len(local_paths)}: {path.name} | rows={pf.metadata.num_rows:,}')
            for rg in range(pf.num_row_groups):
                df = pf.read_row_group(rg).to_pandas()
                columns_seen.update(df.columns)
                if 'prompt' not in df.columns or 'response' not in df.columns:
                    raise ValueError(f'Expected prompt/response columns not found in {path.name}; columns={list(df.columns)}')

                for row in df.itertuples(index=False):
                    row_dict = row._asdict()
                    total_rows_seen += 1
                    prompt = clean_text(row_dict.get('prompt', ''), 900)
                    response = clean_text(row_dict.get('response', ''), 1400)
                    source = clean_text(row_dict.get('source', ''), 120)
                    if not prompt or not response:
                        continue
                    source_counts[source] = source_counts.get(source, 0) + 1
                    record = {
                        'id': indexed_rows,
                        'shard': path.name,
                        'row_group': rg,
                        'prompt': prompt,
                        'response': response,
                        'source': source,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    indexed_rows += 1

                if (rg + 1) % 25 == 0 or rg == pf.num_row_groups - 1:
                    print(f'  row group {rg + 1}/{pf.num_row_groups} | total rows seen {total_rows_seen:,} | indexed {indexed_rows:,}')

    manifest = {
        'repo_id': REPO_ID,
        'dataset_tree_url': DATASET_TREE_URL,
        'files': expected_files_info,
        'parquet_files': parquet_files,
        'columns_seen': sorted(columns_seen),
        'total_rows_seen': total_rows_seen,
        'index_rows': indexed_rows,
        'source_counts_top': sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:30],
        'index_path': str(INDEX_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Done. Indexed {indexed_rows:,} rows into {INDEX_PATH}')


if __name__ == '__main__':
    main()
