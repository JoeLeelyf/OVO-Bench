while true; do
    srun -p mllm --gres=gpu:0 python utils/sample_frames.py \
        --video_dir /mnt/hwfile/mllm/liyifei/OVO-Bench-data/src_videos \
        --chunked_dir /mnt/hwfile/mllm/liyifei/OVO-Bench-data/chunked_videos \
        --sampled_frames_dir /mnt/hwfile/mllm/liyifei/OVO-Bench-data/sampled_frames

    # 检查 Python 文件是否正常退出（0 代表成功退出）
    if [ $? -eq 0 ]; then
        echo "Python 脚本成功执行完成，退出中..."
        break
    else
        # 如果 Python 文件退出时非正常退出（如网络问题），休眠 10 秒
        echo "网络问题或其他错误，等待 10 秒后重试..."
        sleep 10
    fi
done