#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=2

MODEL="mistralai/Mistral-7B-Instruct-v0.2"

TASKS="financial_qa gov_report_summ legal_contract_qa meeting_summ news_summ paper_assistant patent_summ review_summ tv_show_summ"

METRIC="ngram_eval"

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 32k

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 32k --flexicache \
    --num_unstable_heads 0 --rerank_frequency 10000 --topK_budget 64 --unstable_heads_profile_task gov_report

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 32k --flexicache \
    --num_unstable_heads 0 --rerank_frequency 16 --topK_budget 64 --unstable_heads_profile_task gov_report

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 32k --flexicache \
    --num_unstable_heads 64 --rerank_frequency 16 --topK_budget 64 --unstable_heads_profile_task gov_report

python run_benchmark.py \
    --model $MODEL --metric $METRIC --tasks $TASKS --max_length 32k --flexicache \
    --num_unstable_heads 64 --rerank_frequency 16 --topK_budget 128 --unstable_heads_profile_task gov_report