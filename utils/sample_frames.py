from decord import VideoReader, cpu
import argparse
import os
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
import sys
sys.path.append("..")

BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

parser = argparse.ArgumentParser(description='Run OVBench')
parser.add_argument("--anno_path", type=str, default="data/ovo_bench.json", help="Path to the annotations")
parser.add_argument("--video_dir", type=str, default="data/src_videos", help="Root directory of source videos")
parser.add_argument("--chunked_dir", type=str, default="data/chunked_videos", help="Root directory of chunked videos")
parser.add_argument("--sampled_frames_dir", type=str, default="data/sampled_frames", help="Root dir to save sampled frames")
args = parser.parse_args()

def load_video(video_path, max_frames_num=64):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
        
    end_frame = total_frame_num
    if total_frame_num > max_frames_num:
        max_frames_num = max_frames_num
    elif total_frame_num < max_frames_num:
        max_frames_num = total_frame_num - 2
        
    uniform_sampled_frames = np.linspace(0, end_frame - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx)
    spare_frames = spare_frames.asnumpy()

    return spare_frames

with open(args.anno_path, "r") as file:
    data = json.load(file)

for i in tqdm(range(len(data))):
    if data[i]["task"] in BACKWARD_TASKS or data[i]["task"] in REAL_TIME_TASKS:
        chunked_video_path = os.path.join(args.chunked_dir, f"{data[i]['id']}.mp4")
        output_dir = os.path.join(args.sampled_frames_dir, f"{data[i]['id']}")

    elif data[i]["task"] in FORWARD_TASKS:
        for j in range(len(data[i]["test_info"])):
            chunked_video_path = os.path.join(args.chunked_dir, f"{data[i]['id']}.mp4")
            output_dir = os.path.join(args.sampled_frames_dir, f"{data[i]['id']}_{j}")
    
    os.makedirs(output_dir, exist_ok=True)
    assert os.path.exists(chunked_video_path)

    spare_frames = load_video(chunked_video_path)
    for j in range(len(spare_frames)):
        save_path = os.path.join(output_dir, f"{j}.jpg")
        if os.path.exists(save_path):
            print(f"Sampled frames path {save_path} exists. Pass.")
        else:
            # Save sampled frames to path
            img = Image.fromarray(spare_frames[j])
            img.save(save_path)