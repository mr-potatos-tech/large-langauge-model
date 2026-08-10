import json
import re
import sys
from pathlib import Path

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "as", "by", "it", "this", "that", "from", "at", "your", "you", "i", "me", "my", "we", "our",
    "can", "could", "would", "should", "do", "does", "did", "what", "how", "why", "write", "explain", "tell",
    "about", "give", "make", "list", "describe", "please", "help"
}

def words(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())

def query_terms(query):
    terms = [w for w in words(query) if w not in STOPWORDS and len(w) > 2]
    return terms or words(query)

def score_record(query_term_set, prompt, response):
    prompt_words = set(words(prompt))
    response_words = set(words(response[:700]))
    prompt_hits = len(query_term_set & prompt_words)
    response_hits = len(query_term_set & response_words)
    if prompt_hits == 0 and response_hits == 0:
        return 0.0
    length_penalty = 1.0 / (1.0 + max(0, len(prompt) - 260) / 1000)
    return (3.2 * prompt_hits + 0.9 * response_hits) * length_penalty

class DatasetAssistant:
    def __init__(self, export_path):
        self.export_path = Path(export_path)
        self.index_paths = []
        for name in [
            'gpt4all_full_dataset_index.jsonl',
            'ultrachat_full_index.jsonl',
            'ultrachat_sample_index.jsonl',
        ]:
            path = self.export_path / name
            if path.exists():
                self.index_paths.append(path)
        self.knowledge_path = self.export_path / 'assistant_knowledge.json'
        self.small_answers = []
        self.knowledge = {}
        if self.knowledge_path.exists():
            try:
                self.knowledge = json.loads(self.knowledge_path.read_text(encoding='utf-8'))
                for item in self.knowledge.get('qa_pairs', []):
                    for q in item.get('questions', []):
                        self.small_answers.append((str(q).lower(), item.get('answer', '')))
            except Exception:
                pass
    def small_answer(self, message):
        msg_terms = set(query_terms(message))
        best = (0, None)
        for q, ans in self.small_answers:
            score = len(msg_terms & set(query_terms(q)))
            if score > best[0]:
                best = (score, ans)
        return best[1] if best[0] >= 2 else None
    def search_indexes(self, query, top_k=1, max_response_chars=1000):
        q_set = set(query_terms(query))
        best = []
        scanned_total = 0
        for index_path in self.index_paths:
            with index_path.open('r', encoding='utf-8') as f:
                for line in f:
                    scanned_total += 1
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    score = score_record(q_set, rec.get('prompt', ''), rec.get('response', ''))
                    if score <= 0:
                        continue
                    # Prefer UltraChat slightly for general chat/dialogue queries.
                    if rec.get('dataset') == 'openbmb/UltraChat' or rec.get('source') == 'openbmb/UltraChat':
                        score += 0.35
                    score += min(0.5, 100 / (len(rec.get('prompt', '')) + 1000))
                    item = (score, rec, index_path.name)
                    if len(best) < top_k:
                        best.append(item); best.sort(key=lambda x: x[0])
                    elif score > best[0][0]:
                        best[0] = item; best.sort(key=lambda x: x[0])
        best = sorted(best, key=lambda x: x[0], reverse=True)
        results = []
        for score, rec, index_name in best:
            resp = rec.get('response', '')
            if len(resp) > max_response_chars:
                cut = resp.rfind('.', 0, max_response_chars)
                resp = resp[:cut + 1 if cut > 120 else max_response_chars].strip()
            results.append({'score': score, 'record': rec, 'index': index_name, 'response': resp, 'scanned': scanned_total})
        return results
    def ask(self, message):
        message = str(message).strip()
        if not message:
            return 'Please type a non-empty message.'
        results = self.search_indexes(message, top_k=1)
        if results:
            r = results[0]
            rec = r['record']
            dataset = rec.get('dataset') or rec.get('source', 'dataset')
            return (
                f"Dataset answer (searched {r['scanned']:,} rows, dataset={dataset}, index={r['index']}, id={rec.get('id')}):\n"
                f"Matched prompt: {rec.get('prompt','')}\n\n"
                f"Answer: {r['response']}"
            )
        ans = self.small_answer(message)
        if ans:
            return ans
        available = ', '.join(p.name for p in self.index_paths) or 'no local dataset index found'
        return f"I did not find a strong match. Available indexes: {available}. Run `python prepare_dataset.py --include-ultrachat` to build local dataset indexes."

def load_assistant(export_path=None):
    return DatasetAssistant(Path(__file__).parent if export_path is None else export_path)

if __name__ == '__main__':
    assistant = load_assistant(Path(__file__).parent)
    if len(sys.argv) > 1:
        print(assistant.ask(' '.join(sys.argv[1:])))
    else:
        print('Dataset-backed NumPy assistant with GPT4All-J + UltraChat support.')
        print("Type 'exit', 'quit', or 'bye' to stop.")
        print('Indexes:', ', '.join(p.name for p in assistant.index_paths) or 'none found')
        while True:
            try:
                msg = input('You: ').strip()
            except EOFError:
                print('\nGoodbye.')
                break
            if msg.lower() in {'exit', 'quit', 'bye'}:
                print('Model: goodbye')
                break
            if msg:
                print('Model:', assistant.ask(msg))
