srun -p mllm --quotatype=spot --gres=gpu:4 python inference.py \
    --mode offline \
    --task REC \
    --model LLaVA_Video \
    --model_path /path/to/LLaVA-Video-7B-Qwen2
