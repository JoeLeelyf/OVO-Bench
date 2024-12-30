srun -p mllm --gres=gpu:2 python reinference.py \
    --mode offline \
    --task EPM ASI HLD STU OJR ATR ACR OCR FPD REC SSR CRR \
    --model InternVL2 \
    --model_path /PATH/TO/YOUR/MODEL