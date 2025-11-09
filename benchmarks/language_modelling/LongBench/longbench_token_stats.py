#!/usr/bin/env python3
# longbench_token_stats.py
import argparse, csv, json, math
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

STANDARD_TASKS = [
    "narrativeqa","qasper","multifieldqa_en","hotpotqa","2wikimqa","musique",
    "gov_report","qmsum","multi_news","trec","triviaqa","samsum",
    "passage_count","passage_retrieval_en","lcc","repobench-p",
]

# LongBench convention: these tasks generally work better without chat-wrapping;
# we mirror that so prompt token counts match your usual builds.
NO_CHAT_WRAP = {"trec","triviaqa","samsum","lsht","lcc","repobench-p"}

def percentile(vals, p):
    # linear interpolation percentile (p in [0,1]); returns float
    if not vals:
        return 0.0
    a = sorted(vals)
    if len(a) == 1:
        return float(a[0])
    idx = (len(a) - 1) * p
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return float(a[lo])
    return a[lo] + (a[hi] - a[lo]) * (idx - lo) / (hi - lo)

def stats(vals):
    if not vals:
        return (0, 0, 0.0, 0.0, 0.0, 0.0)
    mn = min(vals)
    mx = max(vals)
    p50 = percentile(vals, 0.50)
    avg = sum(vals) / len(vals)
    p75 = percentile(vals, 0.75)
    p90 = percentile(vals, 0.90)
    return (mn, mx, p50, avg, p75, p90)

def flatten_answers(ans):
    # turns nested lists / scalars into a flat list of strings
    if isinstance(ans, (list, tuple)):
        out = []
        for x in ans:
            out.extend(flatten_answers(x))
        return out
    return [str(ans)]

def maybe_apply_chat_template(tokenizer, base_prompt, model_id, dataset_name):
    # Only apply chat template for chat models (e.g., Llama-3 Instruct) and not for NO_CHAT_WRAP tasks
    if dataset_name in NO_CHAT_WRAP:
        return base_prompt
    name = model_id.lower()
    try:
        if "llama-3" in name or "meta-llama-3" in name or "instruct" in name:
            messages = [{"role":"user","content": base_prompt}]
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        pass
    return base_prompt  # default: raw prompt

def main():
    ap = argparse.ArgumentParser(description="Compute token-length stats for LongBench prompts and ground truths.")
    ap.add_argument("--model", required=True, help="HF model ID or local path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    ap.add_argument("--dataset2prompt", required=True, help="Path to dataset2prompt.json (templates with {fields})")
    ap.add_argument("--out_csv", default="longbench_token_stats.csv", help="Output CSV path")
    ap.add_argument("--only", nargs="*", default=None, help="Optional subset of task names to run (from standard set)")
    args = ap.parse_args()

    d2p = json.load(open(args.dataset2prompt, "r"))
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    tasks = args.only if args.only else STANDARD_TASKS

    # Prepare CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Task",
        # Prompt stats
        "Prompt_Min","Prompt_Max","Prompt_P50","Prompt_Avg","Prompt_P75","Prompt_P90",
        # Ground truth (smallest) stats
        "GTmin_Min","GTmin_Max","GTmin_P50","GTmin_Avg","GTmin_P75","GTmin_P90",
        # Ground truth (largest) stats
        "GTmax_Min","GTmax_Max","GTmax_P50","GTmax_Avg","GTmax_P75","GTmax_P90",
    ]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for task in tqdm(tasks, desc="Tasks", unit="task"):
            # Load dataset (non-E, test split)
            ds = load_dataset("THUDM/LongBench", task, split="test")

            prompt_lengths = []
            gt_min_lengths = []
            gt_max_lengths = []

            tmpl = d2p[task]

            for ex in ds:
                # Build base prompt string via template
                try:
                    base_prompt = tmpl.format(**ex)
                except KeyError:
                    # Fallback: use commonly present fields if template keys mismatch
                    # (rare, but keeps script robust/minimal)
                    qs = ex.get("input") or ex.get("question") or ""
                    ctx = ex.get("context") or ex.get("passage") or ""
                    base_prompt = f"{qs}\n\n{ctx}".strip()

                # Optionally wrap in chat template for chatty models
                prompt_text = maybe_apply_chat_template(tok, base_prompt, args.model, task)

                # Token counts
                prompt_len = len(tok.encode(prompt_text))
                prompt_lengths.append(prompt_len)

                answers = flatten_answers(ex.get("answers", []))
                if not answers:
                    # If a sample truly has no answer, count as zero-length ground truth
                    gt_min_lengths.append(0)
                    gt_max_lengths.append(0)
                    continue

                ans_lens = [len(tok.encode(a)) for a in answers]
                gt_min_lengths.append(min(ans_lens))
                gt_max_lengths.append(max(ans_lens))

            p_stats  = stats(prompt_lengths)
            gmin     = stats(gt_min_lengths)
            gmax     = stats(gt_max_lengths)

            row = [task] + [f"{x:.2f}" if isinstance(x, float) and not x.is_integer() else int(x) for x in (
                p_stats[0], p_stats[1], p_stats[2], p_stats[3], p_stats[4], p_stats[5],
                gmin[0], gmin[1], gmin[2], gmin[3], gmin[4], gmin[5],
                gmax[0], gmax[1], gmax[2], gmax[3], gmax[4], gmax[5],
            )]
            writer.writerow(row)

    print(f"Wrote stats to {args.out_csv}")

if __name__ == "__main__":
    main()
