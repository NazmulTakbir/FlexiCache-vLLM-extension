#!/usr/bin/env bash

models=(
  "Mistral-7B-Instruct-v0.2"
  "Qwen2.5-32B-Instruct"
  "Meta-Llama-3.1-8B-Instruct"
  "Mistral-Small-24B-Instruct-2501"
)

declare -A model_to_M=(
  ["Mistral-7B-Instruct-v0.2"]=64
  ["Qwen2.5-32B-Instruct"]=128
  ["Meta-Llama-3.1-8B-Instruct"]=64
  ["Mistral-Small-24B-Instruct-2501"]=80
)

topK_values=(128 64)

for model in "${models[@]}"; do
  M="${model_to_M[$model]}"
  for topK in "${topK_values[@]}"; do
    echo "Model: $model  M: $M  topK: $topK"
    python analyze_head_stability.py \
      --data_root Data-Sorted \
      --model "$model" \
      --window_len 16 \
      --M "$M" \
      --topK "$topK"
  done
done
