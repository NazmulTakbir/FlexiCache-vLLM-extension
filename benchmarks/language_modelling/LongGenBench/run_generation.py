import argparse
from vllm import LLM, SamplingParams
import json
import time
from transformers import AutoTokenizer
import os

model_to_path = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM with command line arguments.')
    parser.add_argument('--model', type=str, default=None, choices=[
        "mistral-7b", "llama-8b"
    ])
    parser.add_argument("--num_samples", type=int, default=None, help="Num of samples to eval on", required=True)
    parser.add_argument('--input_file', type=str, required=True, help='input file path.')
    parser.add_argument('--max_length', type=int, default=16000, help='Maximum length of generation.')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--compress_args_path', type=str, default=None, help='Path to compression arguments file.')

    args = parser.parse_args()
    return args

def process_output(output: str) -> dict:
    blocks = output.split('#*#')
    word_count = len(output.split())
    return {"blocks": blocks, "word_count": word_count}

# Combine inputs, results and word counts and save them
def process_and_save_results(inputs: list, results: list, filename: str) -> None:
    combined = []
    for input_data, result_data in zip(inputs, results):
        combined.append({
            "input": input_data["prompt"],
            "checks_once": input_data["checks_once"],
            "checks_range": input_data["checks_range"],
            "checks_periodic": input_data["checks_periodic"],
            "type": input_data["type"],
            "number": input_data['number'],
            "output_blocks": result_data["blocks"],
            "word_count": result_data["word_count"]  # Adding word count here
        })
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=4)

def setup_compression(args):
    if args.compress_args_path:
        compression_description = args.compress_args_path
        VLLM_BASE = os.environ.get("VLLM_BASE", "-1")
        if VLLM_BASE == "-1":
            raise ValueError("Please set the VLLM_BASE environment variable to the base path of this repository.")
        args.compress_args_path = \
            f"{VLLM_BASE}/benchmarks/language_modelling/LongBench/config/compress_args/{compression_description}.json"
        os.environ["VLLM_COMPRESSION_CONFIG"] = args.compress_args_path

def main():
    args = parse_args()
    setup_compression(args)

    model_path = model_to_path[args.model]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=1, max_tokens=args.max_length, seed=42
    )
    llm = LLM(
        model=model_path, tensor_parallel_size=args.gpu, quantization=None,
        enable_flexicache = True if args.compress_args_path else False,
        enable_prefix_caching=False,
        disable_cascade_attn=True, gpu_memory_utilization=0.95, seed=42
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        inputs = json.load(f)
    prompts = []; inputs_used = []; results = []
    num_week = 0; num_floor = 0; num_menu = 0; num_block = 0
    for input_data in inputs:
        if input_data['type']=="Week" and num_week>=args.num_samples or \
        input_data['type']=="Floor" and num_floor>=args.num_samples or \
        input_data['type']=="Menu Week" and num_menu>=args.num_samples or \
        input_data['type']=="Block" and num_block>=args.num_samples :
            continue
        else:
            if input_data['type']=="Week": num_week+=1
            elif input_data['type']=="Floor": num_floor+=1
            elif input_data['type']=="Menu Week": num_menu+=1
            elif input_data['type']=="Block": num_block+=1
        prompts.append(input_data['prompt'])
        inputs_used.append(input_data)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_time = time.time() - start_time

    fout = open(f"pred/{args.model}_generations.jsonl", 'w', encoding='utf-8')
    total_generated = 0
    for idx, output in enumerate(outputs):
        input_data = inputs_used[idx]
        input_data['response'] = output.outputs[0].text
        results.append(process_output(input_data['prefix'] + input_data['response']))

        fout.write(json.dumps(input_data, ensure_ascii=False)+'\n')
        fout.flush()
        total_generated += len(output.outputs[0].token_ids)

    print(f"\nGenerated {total_generated} tokens in {elapsed_time:.2f}s, Throughput: {total_generated/elapsed_time:.2f} tokens/s")

    output_file = f"pred/{args.model}.json"
    process_and_save_results(inputs_used, results, output_file)
    print(f"\nSaved result to {output_file}")

if __name__ == '__main__':
    main()