#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_USE_V1=1

MODEL="meta-llama/Llama-3.1-8B-Instruct"

TASKS="financial_qa gov_report_summ legal_contract_qa meeting_summ news_summ paper_assistant patent_summ review_summ tv_show_summ"

METRIC="ngram_eval"

python run_benchmark.py --model $MODEL --metric $METRIC --tasks $TASKS --max_length 128k

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 128k --flexicache \
    --num_unstable_heads 0 --rerank_frequency 10000 --topK_budget 128 --unstable_heads_profile_task govt_report

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 128k --flexicache \
    --num_unstable_heads 0 --rerank_frequency 16 --topK_budget 128 --unstable_heads_profile_task govt_report

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 128k --flexicache \
    --num_unstable_heads 64 --rerank_frequency 10000 --topK_budget 128 --unstable_heads_profile_task govt_report

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 128k --flexicache \
    --num_unstable_heads 64 --rerank_frequency 16 --topK_budget 128 --unstable_heads_profile_task govt_report

cd LEval

BASE_DIR="Predictions/$METRIC"
MODEL_PREFIX="$(basename "$MODEL")"

for dir in "$BASE_DIR"/"$MODEL_PREFIX"*; do
    export LEVAL_OUTPUT_DIR="$(basename "$dir")"
    files=( "$dir"/*.pred.jsonl )
    for f in "${files[@]}"; do
        python Evaluation/auto_eval.py --pred_file "$f"
    done
done