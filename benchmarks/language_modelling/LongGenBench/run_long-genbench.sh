export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [ -d "pred" ]; then
    rm -rf pred
fi
mkdir pred

COMPRESS_ARG="topk-256-50-10000-50"

python run_generation.py \
    --model llama-8b \
    --input_file Dataset/DatasetShort.json \
    --num_samples 1 \
    --gpu 1
    # --compress_args_path "$COMPRESS_ARG"

python eval.py \
    --data pred/llama-8b.json \
    --csv pred/lgb_eval.csv \
    --gpu 1