srun -p mllm --gres=gpu:0 --quotatype=spot python inference.py \
    --mode offline \
    --task HLD \
    --model Gemini \
    --gemini_project project-*
