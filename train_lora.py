
"""
Optional real fine-tuning with LoRA on local JSONL dataset indexes.

This is the actual training path, unlike returning memorized rows. It fine-tunes a seq2seq
instruction model on prompt/response pairs and saves adapters to numpy_tiny_llm_export/lora_adapter.

Example:
    pip install -r requirements.txt
    python train_lora.py --max-examples 20000 --epochs 1

Then use a larger custom assistant loader if you want to load the adapter. The default
smart_assistant.py already works zero-shot with FLAN-T5 + semantic context.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

EXPORT_DIR = Path('numpy_tiny_llm_export')
INPUT_FILES = [
    EXPORT_DIR / 'ultrachat_sample_index.jsonl',
    EXPORT_DIR / 'ultrachat_full_index.jsonl',
    EXPORT_DIR / 'gpt4all_full_dataset_index.jsonl',
]


def load_examples(max_examples):
    rows = []
    for path in INPUT_FILES:
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                if max_examples and len(rows) >= max_examples:
                    return rows
                rec = json.loads(line)
                p = str(rec.get('prompt', '')).strip()
                r = str(rec.get('response', '')).strip()
                if p and r:
                    rows.append({'input': 'Answer the user question: ' + p, 'target': r})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='google/flan-t5-small')
    parser.add_argument('--max-examples', type=int, default=20000)
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    examples = load_examples(args.max_examples)
    if not examples:
        raise SystemExit('No examples found. Run prepare_dataset.py first.')
    ds = Dataset.from_list(examples).train_test_split(test_size=0.02, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16, lora_dropout=0.05)
    model = get_peft_model(model, peft_config)

    def preprocess(batch):
        model_inputs = tokenizer(batch['input'], max_length=512, truncation=True)
        labels = tokenizer(text_target=batch['target'], max_length=256, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds['train'].column_names)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(EXPORT_DIR / 'lora_training_output'),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        predict_with_generate=False,
        fp16=False,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    adapter_dir = EXPORT_DIR / 'lora_adapter'
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f'Saved LoRA adapter to {adapter_dir}')


if __name__ == '__main__':
    main()
