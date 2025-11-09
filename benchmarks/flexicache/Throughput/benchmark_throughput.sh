#!/usr/bin/env bash

################################## Variables ##################################

ENABLE_FLEXICACHE=true
TOP_K=128

NUM_PROMPT=500
INPUT_LEN=30000
# OUTPUT_LENS=(50 100 250 500 750 1000 1250 1500)
OUTPUT_LENS=(100 250 500 750 1000 1250 1500)
# OUTPUT_LENS=(50 100 250 500 1000)
# OUTPUT_LENS=(750 1250 1500)

# model="meta-llama/Llama-3.1-8B-Instruct"
model="mistralai/Mistral-7B-Instruct-v0.2"

############################## Running Benchmark ##############################

ratio_in=0.3333
ratio_out=1

leval_base="/srv/m2/ntakbir/LLM-Inference/vllm/vllm-0.8.2/benchmarks/language_modelling/L-Eval"
if [[ "$model" == "mistralai/Mistral-7B-Instruct-v0.2" ]]; then
  dataset_path="${leval_base}/prompts-Mistral-7B-Instruct-v0.2.json"
elif [[ "$model" == "meta-llama/Llama-3.1-8B-Instruct" ]]; then
  dataset_path="${leval_base}/prompts-Llama-3.1-8B-Instruct.json"
else
  echo "Need to generate data for model: $MODEL_NAME"
  exit 1
fi

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND="TRITON_ATTN_VLLM_V1"

VLLM_BASE="/srv/m2/ntakbir/LLM-Inference/vllm/vllm-0.8.2"

OUT_DIR="/srv/m2/ntakbir/LLM-Inference/vllm/vllm-0.8.2/benchmarks/flexicache/Throughput/Results"

for OUTPUT_LEN in "${OUTPUT_LENS[@]}"; do
  OUT_FILE="${OUT_DIR}/FC-${ENABLE_FLEXICACHE}-${model##*/}-${INPUT_LEN}-${OUTPUT_LEN}-${NUM_PROMPT}-${TOP_K}.json"

  CMD=(
    python "${VLLM_BASE}/benchmarks/benchmark_throughput.py" \
      --dataset-name leval \
      --dataset-path "$dataset_path" \
      --model "$model" \
      --gpu-memory-utilization 0.95 \
      --tensor-parallel-size 1 \
      --no-enable-prefix-caching \
      --disable-cascade-attn \
      --max-num-batched-tokens 32768 \
      --max-num-seqs 64 \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --num-prompts "$NUM_PROMPT" \
      --random-range-ratio-input "$ratio_in" \
      --random-range-ratio-output "$ratio_out" \
      --seed 42 \
      --output-json "$OUT_FILE" \
      --rerank-frequency 16 \
      --topK-budget $TOP_K \
      --num-unstable-heads 64 \
      --unstable_heads_profile_task gov_report
  )

  if [ "$ENABLE_FLEXICACHE" = true ]; then
    CMD+=(--enable-flexicache)
  fi

  numactl --cpunodebind=0 --membind=0 "${CMD[@]}"
done