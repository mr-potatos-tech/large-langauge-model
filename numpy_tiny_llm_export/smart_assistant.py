
"""
Smart assistant with two retrieval backends:
- Fast TF-IDF fallback: quick CPU index build, no transformer embedding timeout.
- SentenceTransformer + FAISS: stronger semantic matching when you have time/compute.

Build the fast index:
    python build_vector_index.py --max-records 2000 --backend tfidf

Build the neural index:
    python build_vector_index.py --max-records 5000 --backend sentence-transformers --batch-size 32
"""

import argparse
import json
from pathlib import Path

EXPORT_DIR = Path(__file__).parent
DEFAULT_GENERATOR_MODEL = 'google/flan-t5-small'
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'


class SmartAssistant:
    def __init__(self, export_dir=EXPORT_DIR, generator_model=DEFAULT_GENERATOR_MODEL, embedding_model=None, top_k=4):
        self.export_dir = Path(export_dir)
        self.top_k = top_k
        manifest_path = self.export_dir / 'vector_index_manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError('Missing index manifest. Run: python build_vector_index.py --max-records 2000 --backend tfidf')
        self.manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        self.backend = self.manifest.get('backend', 'sentence-transformers')
        self.metadata = []
        with (self.export_dir / self.manifest.get('metadata_path', 'vector_metadata.jsonl')).open('r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line))

        self.embedder = None
        self.index = None
        self.vectorizer = None
        self.nn = None
        self.matrix = None

        if self.backend == 'tfidf':
            import joblib
            bundle = joblib.load(self.export_dir / self.manifest.get('index_path', 'tfidf_index.joblib'))
            self.vectorizer = bundle['vectorizer']
            self.nn = bundle['nn']
            self.matrix = bundle['matrix']
        else:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.embedding_model_name = embedding_model or self.manifest.get('embedding_model', DEFAULT_EMBEDDING_MODEL)
            self.embedder = SentenceTransformer(self.embedding_model_name, device='cpu')
            self.index = faiss.read_index(str(self.export_dir / self.manifest.get('index_path', 'vector_index.faiss')))

        # Load generator lazily only when answering. If transformers/model download fails, use extractive synthesis.
        self.generator_model = generator_model
        self.generator = None

    def _load_generator(self):
        if self.generator is not None:
            return self.generator
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained(self.generator_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.generator_model)
            self.generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
        except Exception as e:
            self.generator = False
            self.generator_error = str(e)
        return self.generator

    def retrieve(self, question):
        if self.backend == 'tfidf':
            q = self.vectorizer.transform([question])
            distances, ids = self.nn.kneighbors(q, n_neighbors=min(self.top_k, len(self.metadata)))
            results = []
            for dist, idx in zip(distances[0], ids[0]):
                rec = dict(self.metadata[int(idx)])
                rec['score'] = float(1.0 - dist)
                results.append(rec)
            return results
        else:
            q = self.embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype('float32')
            scores, ids = self.index.search(q, self.top_k)
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                rec = dict(self.metadata[int(idx)])
                rec['score'] = float(score)
                results.append(rec)
            return results

    def build_prompt(self, question, contexts):
        context_blocks = []
        for i, rec in enumerate(contexts, start=1):
            context_blocks.append(
                f"Context {i}\nUser example: {rec.get('prompt','')}\nAssistant example: {rec.get('response','')}"
            )
        return (
            'You are a helpful AI assistant. Use the context for facts and style, but do not copy blindly. '
            'Synthesize a direct answer to the new user question.\n\n'
            + '\n\n'.join(context_blocks)
            + f'\n\nNew user question: {question}\n\nFresh answer:'
        )

    def fallback_synthesis(self, question, contexts):
        if not contexts:
            return "I do not have enough context to answer that well."
        best = contexts[0]
        response = best.get('response', '').strip()
        if len(response) > 900:
            cut = response.rfind('.', 0, 900)
            response = response[:cut + 1 if cut > 200 else 900].strip()
        return response

    def ask(self, question, show_sources=True):
        question = str(question).strip()
        if not question:
            return 'Please type a non-empty question.'
        contexts = self.retrieve(question)
        generator = self._load_generator()
        if generator:
            prompt = self.build_prompt(question, contexts)
            answer = generator(prompt, max_new_tokens=180, do_sample=False, num_beams=2, truncation=True)[0]['generated_text'].strip()
        else:
            answer = self.fallback_synthesis(question, contexts)
            answer += '\n\n(Note: generator model was unavailable, so this used retrieved-context synthesis.)'
        if not show_sources:
            return answer
        source_lines = []
        for i, rec in enumerate(contexts[:3], start=1):
            source_lines.append(
                f"{i}. score={rec.get('score', 0):.3f} dataset={rec.get('dataset','')} source_id={rec.get('source_id')} prompt={rec.get('prompt','')[:160]}"
            )
        return answer + f'\n\nRetrieval backend: {self.backend}\nSources used:\n' + '\n'.join(source_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question', nargs='*')
    parser.add_argument('--generator-model', default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument('--embedding-model', default=None)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--no-sources', action='store_true')
    args = parser.parse_args()
    assistant = SmartAssistant(generator_model=args.generator_model, embedding_model=args.embedding_model, top_k=args.top_k)
    if args.question:
        print(assistant.ask(' '.join(args.question), show_sources=not args.no_sources))
    else:
        print('Smart assistant. Type exit/quit/bye to stop.')
        while True:
            try:
                q = input('You: ').strip()
            except EOFError:
                print('\nGoodbye.')
                break
            if q.lower() in {'exit', 'quit', 'bye'}:
                print('Model: goodbye')
                break
            if q:
                print('Model:', assistant.ask(q))


if __name__ == '__main__':
    main()
