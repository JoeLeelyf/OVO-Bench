import argparse
import os
import json
from moviepy.editor import VideoFileClip
import sys
sys.path.append("..")
import math
from tqdm import tqdm

BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

parser = argparse.ArgumentParser(description="Chunk_Video")
parser.add_argument("--anno_path", type=str, default="data/ovo_bench_new.json", help="Path to the annotations")
parser.add_argument("--video_dir", type=str, default="data/src_videos", help="Root directory of source videos")
parser.add_argument("--output_dir", type=str, default="data/chunked_videos", help="Root directory to save the chunked videos")

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

with open(args.anno_path, "r") as file:
    data = json.load(file)

for i in tqdm(range(len(data))):
    if not (data[i]["task"] in FORWARD_TASKS):
        continue
    if data[i]["task"] in BACKWARD_TASKS or data[i]["task"] in REAL_TIME_TASKS:
        output_path = os.path.join(args.output_dir, f"{data[i]['id']}.mp4")
        end_time = math.ceil(data[i]["realtime"])
        if os.path.exists(output_path):
            print(f"Chunked video path {output_path} exists. Pass.")

        if True:
            video = VideoFileClip(os.path.join(args.video_dir, data[i]["video"]))
            video_duration = video.duration
            if end_time > video_duration:
                end_time = video_duration
            clip = video.subclip(0, end_time)
            clip.write_videofile(output_path)

            video.close()
    elif data[i]["task"] in FORWARD_TASKS:
        for j in range(len(data[i]["test_info"])):
            output_path = os.path.join(args.output_dir, f"{data[i]['id']}_{j}.mp4")
            end_time = math.ceil(data[i]["test_info"][j]["realtime"])
            
            if os.path.exists(output_path):
                print(f"Chunked video path {output_path} exists. Pass.")

            if True:
                video = VideoFileClip(os.path.join(args.video_dir, data[i]["video"]))
                video_duration = video.duration
                if end_time > video_duration:
                    end_time = video_duration
                clip = video.subclip(0, end_time)
                clip.write_videofile(output_path)

                video.close()
    
    

