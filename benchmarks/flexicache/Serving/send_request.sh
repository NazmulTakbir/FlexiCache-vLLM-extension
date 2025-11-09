#!/bin/bash

################################## Variables ##################################


# RATES=(0.2 0.3 0.4 0.5 0.6)
RATES=(0.45 0.35 0.25)

ENABLE_FLEXICACHE=false

MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="mistralai/Mistral-7B-Instruct-v0.2"

INPUT_LEN=30000
OUTPUT_LEN=2000
NUM_REQ=500

############################## Running Benchmark ##############################

MODEL_NAME="${MODEL##*/}"

leval_base="/srv/m2/ntakbir/LLM-Inference/vllm/vllm-0.8.2/benchmarks/language_modelling/L-Eval"
if [[ "$MODEL_NAME" == "Mistral-7B-Instruct-v0.2" ]]; then
  dataset_path="${leval_base}/prompts-Mistral-7B-Instruct-v0.2.json"
elif [[ "$MODEL_NAME" == "Llama-3.1-8B-Instruct" ]]; then
  dataset_path="${leval_base}/prompts-Llama-3.1-8B-Instruct.json"
else
  echo "Need to generate data for model: $MODEL_NAME"
  exit 1
fi

VLLM_BASE="/srv/m2/ntakbir/LLM-Inference/vllm/vllm-0.8.2"

OUT_DIR="/srv/m2/ntakbir/LLM-Inference/vllm/vllm-0.8.2/benchmarks/flexicache/Serving/Results"

for RATE in "${RATES[@]}"; do
  OUT_FILE="FC-${ENABLE_FLEXICACHE}-${MODEL_NAME}-${INPUT_LEN}-${OUTPUT_LEN}-${NUM_REQ}-${RATE}.json"
  python "${VLLM_BASE}/benchmarks/benchmark_serving.py" \
    --backend vllm \
    --model $MODEL \
    --dataset-name leval \
    --dataset-path "$dataset_path" \
    --random-input-len $INPUT_LEN \
    --random-output-len $OUTPUT_LEN \
    --num-prompts $NUM_REQ \
    --seed 42 \
    --random-range-ratio-input 0.3333 \
    --random-range-ratio-output 0.01 \
    --ignore-eos \
    --save-result \
    --result-filename $OUT_FILE \
    --result-dir $OUT_DIR \
    --request-rate $RATE
done