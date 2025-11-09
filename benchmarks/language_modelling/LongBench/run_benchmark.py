import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from fastchat.model import get_conversation_template

model2path     = json.load(open("config/model2path.json", "r"))
model2maxlen   = json.load(open("config/model2maxlen.json", "r"))
model2revision = json.load(open("config/model2revision.json", "r"))
dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, required=True, choices=[
        "Meta-Llama-3.1-8B-Instruct", "Mistral-Small-24B-Instruct-2501",
        "Qwen2.5-32B-Instruct", "Mistral-7B-Instruct-v0.2"
    ])
    parser.add_argument('--flexicache', action='store_true', default=False, help="Enable Flexicache")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument( '--dataset', type=str, nargs='+', required=True, help="List of datasets to evaluate on")
    parser.add_argument('--batch_size', type=int, default=10, help="How many requests to send in parallel")
    parser.add_argument('--num_unstable_heads', type=int, default=-1, help="Number of unstable heads for Flexicache")
    parser.add_argument('--rerank_frequency', type=int, default=-1, help="Rerank frequency for Flexicache")
    parser.add_argument('--topK_budget', type=int, default=-1, help="TopK budget for Flexicache")
    parser.add_argument('--unstable_heads_profile_task', type=str, default=None,
                        help="Task to profile unstable heads")

    return parser.parse_args()

def get_output_path(dataset, args):
    file_name = \
        f"flexicache-{args.num_unstable_heads}-unstable-{args.rerank_frequency}-rerank-{args.topK_budget}-topK" \
        if args.flexicache else "no_flexicache"

    output_path = \
        f"pred{'_e' if args.e else ''}/{args.model}/{dataset}/{file_name}.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return output_path

def get_data(dataset, e):
    if e:
        datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
        ]
    else:
        datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
        ]

    if dataset not in datasets:
        raise ValueError(f"Dataset {dataset} not found in datasets")

    if e:
        return load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
    else:
        return load_dataset('THUDM/LongBench', dataset, split='test')

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif 'llama-3' in model_name.lower() or 'mistral' in model_name.lower() or 'qwen' in model_name.lower():
        prompt = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False
        )
    return prompt

def get_prompt(model_name, data_sample, dataset_name, tokenizer, max_gen_len):
    prompt = dataset2prompt[dataset_name].format(**data_sample)

    if dataset_name not in [
        "trec",
        "triviaqa",
        "samsum",
        "lsht",
        "lcc",
        "repobench-p",
    ]:  # chat models are better off without build prompts on these tasks
        prompt = build_chat(tokenizer, prompt, model_name)

    # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if "chatglm3" in model_name:
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

    if len(tokenized_prompt) + max_gen_len > model2maxlen[model_name]:
        half = int((model2maxlen[model_name] - max_gen_len) / 2)
        prompt = tokenizer.decode(
            tokenized_prompt[:half], skip_special_tokens=True
        ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

    return prompt

def get_llm_tokenizer(args):
    model_name     = args.model
    model_path     = model2path[model_name]
    model_revision = model2revision.get(model_name)
    tokenizer      = AutoTokenizer.from_pretrained(model_path)

    llm = LLM(
        model_path,
        revision=model_revision,
        enable_flexicache=args.flexicache,
        enable_prefix_caching=False,
        disable_cascade_attn=True,
        gpu_memory_utilization=0.95,
        seed=42,
        max_num_seqs=args.batch_size,
        num_unstable_heads=args.num_unstable_heads,
        rerank_frequency=args.rerank_frequency,
        topK_budget=args.topK_budget,
        unstable_heads_profile_task=args.unstable_heads_profile_task
    )

    return llm, tokenizer

def generate(args, llm, tokenizer, dataset, data, out_path):
    max_gen_len = dataset2maxlen[dataset]
    sampling_params = SamplingParams(
        temperature=0, top_p=1.0, seed=42, max_tokens=max_gen_len
    )

    inputs = []
    for data_sample in data:
        prompt = get_prompt(args.model, data_sample, dataset, tokenizer, max_gen_len)
        inputs.append({"data_sample": data_sample, "prompt": prompt})

    outputs = []
    for i in tqdm(range(0, len(inputs), args.batch_size)):
        batch = inputs[i:i + args.batch_size]
        
        prompts          = [item["prompt"] for item in batch]
        num_input_tokens = [len(tokenizer.encode(prompt)) for prompt in prompts]
        
        data_samples = [item["data_sample"] for item in batch]

        f = open('tmp.txt', "w", encoding="utf-8")
        f.write(prompts[0])
        f.close()
        out = llm.generate(prompts, sampling_params, use_tqdm=False)
        
        preds           = [output.outputs[0].text.strip() for output in out]
        num_pred_tokens = [len(tokenizer.encode(pred)) for pred in preds]

        for i in range(len(out)):
            outputs.append({
                "pred": post_process(preds[i], args.model),
                "answers": data_samples[i]["answers"],
                "all_classes": data_samples[i]["all_classes"],
                "length": data_samples[i]["length"],
                "num_input_tokens": num_input_tokens[i],
                "num_gen_tokens": num_pred_tokens[i],
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for output in outputs:
            json.dump(output, f, ensure_ascii=False)
            f.write("\n")

if __name__ == '__main__':
    args = parse_args()
    llm, tokenizer = get_llm_tokenizer(args)
    for dataset in args.dataset:
        data     = list(get_data(dataset, args.e))
        out_path = get_output_path(dataset, args)
        generate(args, llm, tokenizer, dataset, data, out_path)