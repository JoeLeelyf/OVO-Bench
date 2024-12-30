srun -p mllm --quotatype=spot --gres=gpu:2 python inference.py \
    --mode offline \
    --task EPM ASI HLD STU OJR ATR ACR OCR FPD REC SSR CRR \
    --model QWen2VL_7B \
    --model_path /PATH/TO/YOUR/MODEL