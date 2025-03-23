srun -p mllm --quotatype=spot --gres=gpu:0 python inference.py \
    --mode offline \
    --task SSR \
    --model GPT \
    --gpt_api sk-proj-*