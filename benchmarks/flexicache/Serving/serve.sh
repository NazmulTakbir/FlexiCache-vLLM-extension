#!/bin/bash

export VLLM_ATTENTION_BACKEND="TRITON_ATTN_VLLM_V1"
export VLLM_USE_V1="1"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="mistralai/Mistral-7B-Instruct-v0.2"

ENABLE_FLEXICACHE=false

CMD=(
  vllm serve "$MODEL"
  --gpu-memory-utilization 0.95
  --tensor-parallel-size 1
  --no-enable-prefix-caching
  --disable-cascade-attn
  --max-num-batched-tokens 32768
  --max-num-seqs 64
  --rerank-frequency 16
  --topK-budget 64
  --num-unstable-heads 64
  --unstable_heads_profile_task gov_report
)

if [ "$ENABLE_FLEXICACHE" = true ]; then
  CMD+=(--enable-flexicache)
fi

numactl --cpunodebind=0 --membind=0 "${CMD[@]}"