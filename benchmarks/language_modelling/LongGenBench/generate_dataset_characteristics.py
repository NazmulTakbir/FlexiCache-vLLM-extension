import json
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

path = f"pred/mistral-7b_generations.jsonl"
model = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model)

pred_lens = []
prompt_lens = []
response_lens = []
with open(path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        data = json.loads(line)
        prompt = data['prompt']
        response = data['response']
        prompt_lens.append(len(tokenizer(prompt)["input_ids"]))
        response_lens.append(len(tokenizer(prompt + response)["input_ids"]))

print(f"{'Type':<25} {'Min':>6} {'Max':>6} {'Median':>8} {'Avg':>8} {'P75':>8} {'P90':>8}")
print("-" * 75)

s = {
    "min": np.min(prompt_lens),
    "max": np.max(prompt_lens),
    "median": round(np.median(prompt_lens)),
    "average": round(np.mean(prompt_lens)),
    "p75": round(np.percentile(prompt_lens, 75)),
    "p90": round(np.percentile(prompt_lens, 90))
}
print(f"{'Prompt':<25} {s['min']:>6} {s['max']:>6} {s['median']:>8} {s['average']:>8} {s['p75']:>8} {s['p90']:>8}")

s = {
    "min": np.min(response_lens),
    "max": np.max(response_lens),
    "median": round(np.median(response_lens)),
    "average": round(np.mean(response_lens)),
    "p75": round(np.percentile(response_lens, 75)),
    "p90": round(np.percentile(response_lens, 90))
}
print(f"{'Response':<25} {s['min']:>6} {s['max']:>6} {s['median']:>8} {s['average']:>8} {s['p75']:>8} {s['p90']:>8}")