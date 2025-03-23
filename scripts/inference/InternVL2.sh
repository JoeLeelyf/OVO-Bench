srun -p mllm --quotatype=spot --gres=gpu:4 python inference.py \
    --mode offline \
    --task REC \
    --model InternVL2 \
    --model_path /path/to/InternVL2-8B