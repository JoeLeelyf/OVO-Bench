srun -p mllm --gres=gpu:1 python inference.py \
    --mode offline \
    --task OCR ACR ATR \
    --model MiniCPM_o \
    --model_path /path/to/MiniCPM-o-2_6
