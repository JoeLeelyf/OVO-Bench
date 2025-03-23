srun -p mllm --quotatype=spot --gres=gpu:4 python inference.py \
    --mode offline \
    --task REC \
    --model QWen2VL_7B \
    --model_path /path/to/Qwen2-VL-7B-Instruct
