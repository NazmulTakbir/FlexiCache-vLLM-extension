import os
import torch

model = 'Meta-Llama-3.1-8B-Instruct'
dataset = 'Government-Report'
sample_num = 0

folder = f'{model}/{dataset}/sample-{str(sample_num).zfill(3)}'
files = sorted(os.listdir(folder))

# print(files[:5])
# output: ['decode-step-001-prompt-len-10733.pt', 'decode-step-002-prompt-len-10733.pt', 'decode-step-003-prompt-len-10733.pt', 'decode-step-004-prompt-len-10733.pt', 'decode-step-005-prompt-len-10733.pt'

############################################
# For Meta-Llama-3.1-8B-Instruct Model
# Number of Layers: 32
# Number of KV Heads per Layer: 8
# Top-K Pages per Head: 256
# Page Size: 16 tokens
############################################


def find_overlap_percentage(layer, head, reference_step, num_compare_steps):
    top_k_sets = []
    for file in files:
        tensor = torch.load(os.path.join(folder, file))
        # print(tensor.shape)  # torch.Size([32, 8, 256]) -> [num_layers, num_heads, top_k_pages]
        top_k_sets.append(set(tensor[layer, head, :].tolist()))

    print(f"Printing overlap percentage for layer {layer} head {head} with reference step {reference_step}")
    first_set = top_k_sets[reference_step]
    for i in range(reference_step + 1, reference_step + 1 + num_compare_steps):
        overlap = first_set.intersection(top_k_sets[i])
        overlap_percentage = len(overlap) / len(first_set) * 100
        print(f'Overlap percentage with step {i}: {overlap_percentage:.2f}%')
    print('-' * 50)

find_overlap_percentage(layer=0, head=0, reference_step=0, num_compare_steps=5)

find_overlap_percentage(layer=14, head=7, reference_step=100, num_compare_steps=10)
