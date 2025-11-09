import json
from transformers import AutoTokenizer
import numpy as np

def read_jsonl(train_fn):
    res = []
    with open(train_fn) as f:
        for line in f:
            res.append(json.loads(line))
    return res

model_to_max_lens = {
    "meta-llama/Llama-3.1-8B-Instruct": 131072,
    "mistralai/Mistral-7B-Instruct-v0.2": 32768,
}

for model_name, max_len in model_to_max_lens.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tasks = [
        "financial_qa", "gov_report_summ", "legal_contract_qa", "meeting_summ", "news_summ",
        "paper_assistant", "patent_summ", "review_summ", "tv_show_summ"
    ]

    sys_prompt = \
        "Now you are given a very long document. Please follow the instruction after this document. These instructions may include summarizing a document, answering questions based on the document, or writing a required paragraph. "

    prompts = []
    for task in tasks:
        task_path = f"LEval/LEval-data/Open-ended-tasks/{task}.jsonl"
        data = read_jsonl(task_path)
        for d in data:
            document     = d['input']
            instructions = d['instructions']
            outputs      = d['outputs']

            for inst, out in zip(instructions, outputs):
                messages = [{"role": "system", "content" : sys_prompt}]

                # multiplying by 2 to encourage longer outputs
                suggested_output_length = len(out.split()) * 2
                context = \
                    "Document is as follows. {} Instruction: {} " + f"The suggested output length is around {suggested_output_length} words. Output: "
                messages.append({"role": "user", "content": context.format(document, inst)})

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                prompt_tokens = tokenizer.encode(prompt)
                if len(prompt_tokens) > (max_len - 2000): # subtracting 2000 to leave room for generation
                    half = int((max_len - 2000) / 2)
                    prompt = \
                        tokenizer.decode(prompt_tokens[:half], skip_special_tokens=True) \
                        + tokenizer.decode(prompt_tokens[-half:], skip_special_tokens=True)
                prompts.append(prompt)

    # prompt_lens = np.array([len(tokenizer.encode(p)) for p in prompts])
    # percentiles = np.percentile(prompt_lens, [25, 50, 75, 95])
    # print(percentiles)
    # print(np.mean(prompt_lens))
    # print(np.min(prompt_lens))
    # print(np.max(prompt_lens))

    output_name = f'prompts-{model_name.split("/")[-1]}.json'
    with open(output_name, "w") as f:
        json.dump(prompts, f)