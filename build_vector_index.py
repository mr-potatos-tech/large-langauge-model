
"""
Build a semantic index for the assistant.

Fast notebook-safe usage:
    python build_vector_index.py --max-records 2000 --backend tfidf

Neural embedding usage:
    python build_vector_index.py --max-records 5000 --backend sentence-transformers --batch-size 32

Why this version exists:
The previous notebook command tried to embed 12,000 records with a transformer and timed out.
This script streams records, supports smaller defaults, and provides a TF-IDF fallback that is
much faster on CPU. The assistant can use either the FAISS neural index or the TF-IDF fallback.
"""

import argparse
import json
from pathlib import Path

EXPORT_DIR = Path('numpy_tiny_llm_export')
DEFAULT_ST_MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
INPUT_FILES = [
    EXPORT_DIR / 'ultrachat_sample_index.jsonl',
    EXPORT_DIR / 'ultrachat_full_index.jsonl',
    EXPORT_DIR / 'gpt4all_full_dataset_index.jsonl',
]


def iter_records(max_records=2000):
    seen = 0
    for path in INPUT_FILES:
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                if max_records and seen >= max_records:
                    return
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                prompt = str(rec.get('prompt', '')).strip()
                response = str(rec.get('response', '')).strip()
                if not prompt or not response:
                    continue
                search_text = (prompt + ' ' + response[:350]).strip()[:1600]
                yield {
                    'id': seen,
                    'dataset': rec.get('dataset') or rec.get('source') or '',
                    'source_file': rec.get('file') or rec.get('shard') or '',
                    'source_id': rec.get('id'),
                    'prompt': prompt[:1200],
                    'response': response[:1800],
                    'search_text': search_text,
                }
                seen += 1


def write_metadata(records):
    with (EXPORT_DIR / 'vector_metadata.jsonl').open('w', encoding='utf-8') as f:
        for rec in records:
            out = dict(rec)
            out.pop('search_text', None)
            f.write(json.dumps(out, ensure_ascii=False) + '\n')


def build_tfidf(records):
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        raise SystemExit('TF-IDF backend needs scikit-learn and joblib. Install with: pip install scikit-learn joblib') from e

    texts = [r['search_text'] for r in records]
    print(f'Building fast TF-IDF semantic-ish index for {len(records):,} records ...')
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=80000,
        min_df=1,
        sublinear_tf=True,
        norm='l2',
    )
    matrix = vectorizer.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=min(8, len(records)), metric='cosine', algorithm='brute')
    nn.fit(matrix)
    joblib.dump({'vectorizer': vectorizer, 'nn': nn, 'matrix': matrix}, EXPORT_DIR / 'tfidf_index.joblib')
    manifest = {
        'backend': 'tfidf',
        'records': len(records),
        'index_path': 'tfidf_index.joblib',
        'metadata_path': 'vector_metadata.jsonl',
        'note': 'Fast CPU fallback. For stronger semantic matching, rebuild with --backend sentence-transformers.',
    }
    (EXPORT_DIR / 'vector_index_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def build_sentence_transformers(records, model_name, batch_size):
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit('Sentence-transformers backend needs faiss-cpu, numpy, sentence-transformers.') from e

    texts = [r['search_text'] for r in records]
    print(f'Embedding {len(records):,} records with {model_name} ...')
    model = SentenceTransformer(model_name, device='cpu')
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(EXPORT_DIR / 'vector_index.faiss'))
    manifest = {
        'backend': 'sentence-transformers',
        'embedding_model': model_name,
        'records': len(records),
        'dimension': int(dim),
        'index_path': 'vector_index.faiss',
        'metadata_path': 'vector_metadata.jsonl',
    }
    (EXPORT_DIR / 'vector_index_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['tfidf', 'sentence-transformers'], default='tfidf')
    parser.add_argument('--model', default=DEFAULT_ST_MODEL)
    parser.add_argument('--max-records', type=int, default=2000, help='0 means all local records')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    EXPORT_DIR.mkdir(exist_ok=True)
    records = list(iter_records(max_records=args.max_records))
    if not records:
        raise SystemExit('No records found. Run prepare_dataset.py first, or keep ultrachat_sample_index.jsonl available.')

    write_metadata(records)
    if args.backend == 'tfidf':
        build_tfidf(records)
    else:
        build_sentence_transformers(records, args.model, args.batch_size)
    print(f'Done. Indexed {len(records):,} records using backend={args.backend}.')


if __name__ == '__main__':
    main()
