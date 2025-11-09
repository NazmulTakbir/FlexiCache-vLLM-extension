import json
import os
import argparse
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

from tqdm import tqdm
from LEval.Baselines.LEval_config import (
    k_to_number, build_key_data_pairs, max_new_tokens, get_sys_prompt 
)

tokenizer = None

def num_tokens_from_string(string: str, model_path: str) -> int:
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return len(tokenizer.encode(string))

def get_llm(args):
    llm = LLM(
        args.model,
        enable_flexicache=args.flexicache,
        gpu_memory_utilization=0.95,
        seed=42,
        enable_prefix_caching=False,
        disable_cascade_attn=True,
        max_num_seqs=32,
        num_unstable_heads=args.num_unstable_heads,
        rerank_frequency=args.rerank_frequency,
        topK_budget=args.topK_budget,
        unstable_heads_profile_task=args.unstable_heads_profile_task
    )
    return llm

def generate(args, llm, key_data_pairs):
    global tokenizer

    sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=max_new_tokens)

    for file_name in key_data_pairs:
        sys_prompt = get_sys_prompt(args, file_name)
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while num_tokens_from_string(document, args.model) > max_length:
                if "code" not in file_name:
                    document = " ".join(document.split(" ")[:max_length - cnt]) # chunk the input len from right
                else:
                    document = " ".join(document.split(" ")[cnt - max_length:]) # chunk the input len from left
                cnt += 250
            
            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                messages = [{"role": "system", "content" : sys_prompt}]
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "codeU" in file_name:
                    messages.append({"role": "user", "content": document + "\n\n" + inst})
                    save_d['prompt'] = sys_prompt + inst

                elif args.metric == "exam_eval":
                    context = "Document is as follows. {} Question: {} \nPlease directly give answer without any additional output or explanation\n Answer: "
                    messages.append({"role": "user", "content": context.format(document, inst)})
                    save_d['prompt'] = sys_prompt + context
                else:
                    context = "Document is as follows. {} Instruction: {} " + f"The suggested output length is around {len(out.split())} words. Output: "
                    messages.append({"role": "user", "content": context.format(document, inst)})
                    save_d['prompt'] = sys_prompt + context

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                out = llm.generate([prompt], sampling_params, use_tqdm=False)
                ret = out[0].outputs[0].text.strip()

                save_d[f'{model_name}_pred'] = ret
                save_d['evaluation'] = d['evaluation']

                save_d['num_prompt_tokens'] = num_tokens_from_string(prompt, args.model)
                save_d['num_generated_tokens'] = num_tokens_from_string(ret, args.model)

                # test the factuality in scientific fiction
                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.",
                                                "based on the real-world knowledge and facts up until your last training") + "\nPlease directly give answer without any additional output or explanation \nAnswer:"
                    messages.append({"role": "user", "content": text_inputs})
                    
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    out = llm.generate([prompt], sampling_params, use_tqdm=False)
                    ret = out[0].outputs[0].text.strip()

                    save_d[f'{model_name}_pred'] += f" [fact: {ret}]"

                fw.write(json.dumps(save_d) + '\n')
        fw.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, required=True, choices=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ])
    parser.add_argument('--flexicache', action='store_true', default=False, help="Enable Flexicache")
    parser.add_argument(
        '--metric', choices=["llm_turbo_eval","llm_gpt4_eval","exam_eval", "ngram_eval", "human_eval"],
        default='ngram_eval', help='metric name from ["turbo_eval","gpt4_eval","auto_eval", ...]'
    )
    parser.add_argument('--max_length', default="128k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument(
        '--task_path', type=str, default=None,
        help= 'set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ '
    )
    parser.add_argument(
        '--task_name', type=str, default=None,
        help='optional, if not set, we will test all. set this if you want test a specific task from huggingface, example: coursera, tpo'
    )
    parser.add_argument('--tasks', type=str, default=None, required=True, nargs='+',help="Tasks for FlexiCache")
    parser.add_argument('--mc_tasks', action='store_true')
    parser.add_argument('--num_unstable_heads', type=int, default=-1, help="Number of unstable heads for Flexicache")
    parser.add_argument('--rerank_frequency', type=int, default=-1, help="Rerank frequency for Flexicache")
    parser.add_argument('--topK_budget', type=int, default=-1, help="TopK budget for Flexicache")
    parser.add_argument('--unstable_heads_profile_task', type=str, default=None,
                        help="Task to profile unstable heads")

    return parser.parse_args()

if __name__ == "__main__":
    os.chdir('LEval') # clone the repo first
    
    args = parse_args()

    llm = get_llm(args)
    model_name  = args.model.split('/')[-1]

    for task in args.tasks:
        args.task_path = f"LEval-data/Open-ended-tasks/{task}.jsonl"

        if not args.flexicache:
            file_name = f"{model_name}-no-flexicache"
        else:
            file_name = f"{model_name}-{args.num_unstable_heads}-unstable-{args.rerank_frequency}-rerank-topK-{args.topK_budget}"

        output_path = f"Predictions/{args.metric}/{file_name}"
        
        key_data_pairs = {}
        max_length = k_to_number(args.max_length) - max_new_tokens
        
        build_key_data_pairs(args, key_data_pairs, output_path)

        generate(args, llm, key_data_pairs)
