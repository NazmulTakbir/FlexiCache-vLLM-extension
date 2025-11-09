#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <rank_frequency> <port_number> <gpu_number>"
    exit 1
fi

RANK_FREQ="$1"
PORT="$2"
GPU_NUM="$3"

export CUDA_VISIBLE_DEVICES=$GPU_NUM

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_COMPRESSION_CONFIG=/srv/m2/ntakbir/LLM-Inference/vllm/vllm-0.8.2/benchmarks/language_modelling/LongBench/config/compress_args/topk-256-50-10000-50.json 
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export RANK_FREQUENCY="$RANK_FREQ"

MODEL=meta-llama/Llama-3.1-8B-Instruct

python -m vllm.entrypoints.api_server \
  --model $MODEL \
  --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching \
  --disable-cascade-attn \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --port $PORT \
  --enable-flexicache
