export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
# export VLLM_FLASH_ATTN_VERSION=2
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1

# unset VLLM_ATTENTION_BACKEND            
# unset VLLM_V1_USE_PREFILL_DECODE_ATTENTION

python benchmark_throughput.py \
    --dataset-name random --model meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 --no-enable-prefix-caching --disable-cascade-attn \
    --input-len 10000 --output-len 5000 --num-prompts 10

python benchmark_throughput.py \
    --dataset-name random --model meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 --no-enable-prefix-caching --disable-cascade-attn \
    --input-len 15000 --output-len 5000 --num-prompts 10

python benchmark_throughput.py \
    --dataset-name random --model meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 --no-enable-prefix-caching --disable-cascade-attn \
    --input-len 20000 --output-len 5000 --num-prompts 10

python benchmark_throughput.py \
    --dataset-name random --model meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1 --no-enable-prefix-caching --disable-cascade-attn \
    --input-len 30000 --output-len 5000 --num-prompts 10