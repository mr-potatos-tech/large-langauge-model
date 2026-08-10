import json
import re
import sys
from pathlib import Path
import numpy as np

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "as", "by", "it", "this", "that", "from", "at", "your", "you", "i", "me", "my", "we", "our",
    "can", "could", "would", "should", "do", "does", "did", "what", "how", "why", "write", "explain", "tell",
    "about", "give", "make", "list", "describe"
}

def words(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())

def query_terms(query):
    terms = [w for w in words(query) if w not in STOPWORDS and len(w) > 2]
    return terms or words(query)

def score_record(query_term_set, prompt, response):
    prompt_words = set(words(prompt))
    response_words = set(words(response[:500]))
    prompt_hits = len(query_term_set & prompt_words)
    response_hits = len(query_term_set & response_words)
    if prompt_hits == 0 and response_hits == 0:
        return 0.0
    length_penalty = 1.0 / (1.0 + max(0, len(prompt) - 250) / 1000)
    return (3.0 * prompt_hits + 0.8 * response_hits) * length_penalty

class FullDatasetAssistant:
    def __init__(self, export_path):
        self.export_path = Path(export_path)
        self.index_path = self.export_path / "gpt4all_full_dataset_index.jsonl"
        self.manifest_path = self.export_path / "gpt4all_full_dataset_manifest.json"
        self.knowledge_path = self.export_path / "assistant_knowledge.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
        self.small_answers = []
        if self.knowledge_path.exists():
            try:
                k = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
                for item in k.get("qa_pairs", []):
                    for q in item.get("questions", []):
                        self.small_answers.append((str(q).lower(), item.get("answer", "")))
            except Exception:
                pass
    def small_answer(self, message):
        msg_words = set(query_terms(message))
        best = (0, None)
        for q, ans in self.small_answers:
            score = len(msg_words & set(query_terms(q)))
            if score > best[0]:
                best = (score, ans)
        if best[0] >= 2:
            return best[1]
        return None
    def search_dataset(self, query, top_k=1, max_response_chars=900):
        q_set = set(query_terms(query))
        best = []
        scanned = 0
        if not self.index_path.exists():
            return [], scanned
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                scanned += 1
                rec = json.loads(line)
                score = score_record(q_set, rec.get("prompt", ""), rec.get("response", ""))
                if score <= 0:
                    continue
                score += min(0.5, 100 / (len(rec.get("prompt", "")) + 1000))
                item = (score, rec)
                if len(best) < top_k:
                    best.append(item); best.sort(key=lambda x: x[0])
                elif score > best[0][0]:
                    best[0] = item; best.sort(key=lambda x: x[0])
        best = sorted(best, key=lambda x: x[0], reverse=True)
        results = []
        for score, rec in best:
            resp = rec.get("response", "")
            if len(resp) > max_response_chars:
                cut = resp.rfind(".", 0, max_response_chars)
                resp = resp[:cut + 1 if cut > 120 else max_response_chars].strip()
            results.append({"score": score, "record": rec, "response": resp, "scanned": scanned})
        return results, scanned
    def ask(self, message):
        message = str(message).strip()
        if not message:
            return "Please type a non-empty message."
        results, scanned = self.search_dataset(message, top_k=1)
        if results:
            r = results[0]
            rec = r["record"]
            return (
                f"Dataset answer (searched {scanned:,} rows, source={rec.get('source','')}, shard={rec.get('shard','')}, id={rec.get('id')}):\n"
                f"Matched prompt: {rec.get('prompt','')}\n\n"
                f"Answer: {r['response']}"
            )
        ans = self.small_answer(message)
        if ans:
            return ans
        return f"I searched the full dataset index ({self.manifest.get('index_rows', 0):,} rows) but did not find a strong match."

def load_assistant(export_path=None):
    return FullDatasetAssistant(Path(__file__).parent if export_path is None else export_path)

if __name__ == "__main__":
    assistant = load_assistant(Path(__file__).parent)
    if len(sys.argv) > 1:
        print(assistant.ask(" ".join(sys.argv[1:])))
    else:
        print("Full-dataset GPT4All-J NumPy assistant. Type 'exit', 'quit', or 'bye' to stop.")
        print(f"Using dataset rows: {assistant.manifest.get('index_rows', 0):,}")
        while True:
            try:
                msg = input("You: ").strip()
            except EOFError:
                print("\nGoodbye.")
                break
            if msg.lower() in {"exit", "quit", "bye"}:
                print("Model: goodbye")
                break
            if msg:
                print("Model:", assistant.ask(msg))
