srun -p mllm --quotatype=spot --gres=gpu:1 python inference.py \
    --mode offline \
    --task CRR \
    --model LongVU \
    --model_path /path/to/LongVU_Qwen2_7B
