srun -p mllm --gres=gpu:0 python utils/chunk_videos.py \
    --video_dir /path/to/src_videos \
    --output_dir /path/to/chunked_videos \