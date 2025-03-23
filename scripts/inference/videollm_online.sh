srun -p mllm --gres=gpu:1 python inference.py \
    --mode offline \
    --task OCR \
    --model videollm_online \
    --model_path /path/to/videollm-online-8b-v1plus