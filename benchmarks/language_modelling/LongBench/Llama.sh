#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0

MODEL="Meta-Llama-3.1-8B-Instruct"

BATCH_SIZE=15

DATASETS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique qmsum gov_report multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
python run_benchmark.py \
    --model $MODEL --dataset $DATASETS --batch_size $BATCH_SIZE

DATASETS="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
python run_benchmark.py \
    --model $MODEL --dataset $DATASETS --batch_size $BATCH_SIZE --flexicache \
    --num_unstable_heads 64 --rerank_frequency 16 --topK_budget 64 --unstable_heads_profile_task gov_report

DATASETS="gov_report"
python run_benchmark.py \
    --model $MODEL --dataset $DATASETS --batch_size $BATCH_SIZE --flexicache \
    --num_unstable_heads 64 --rerank_frequency 16 --topK_budget 64 --unstable_heads_profile_task paper_assistant