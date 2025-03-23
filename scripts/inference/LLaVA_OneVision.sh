srun -p mllm --quotatype=spot --gres=gpu:4 python inference.py \
    --mode offline \
    --task SSR CRR \
    --model LLaVA_OneVision \
    --model_path /path/to/llava-onevision-qwen2-7b-ov-chat
