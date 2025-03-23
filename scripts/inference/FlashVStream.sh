srun -p mllm --gres=gpu:1 --quotatype=spot python inference.py \
    --mode offline \
    --task REC \
    --model FlashVStream \
    --model_path /path/to/Flash-VStream-7b