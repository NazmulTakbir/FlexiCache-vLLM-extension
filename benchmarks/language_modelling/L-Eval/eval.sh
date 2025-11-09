#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_USE_V1=1

MODELS="meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2"
METRIC="ngram_eval"

cd LEval

BASE_DIR="Predictions/$METRIC"

for MODEL in $MODELS; do
    MODEL_PREFIX="$(basename "$MODEL")"

    for dir in "$BASE_DIR"/"$MODEL_PREFIX"*; do
        export LEVAL_OUTPUT_DIR="$(basename "$dir")"
        files=( "$dir"/*.pred.jsonl )
        for f in "${files[@]}"; do
            python Evaluation/auto_eval.py --pred_file "$f"
        done
    done
done
