import argparse
import os
import json
from models import *

parser = argparse.ArgumentParser(description='Run OVBench')
parser.add_argument("--anno_path", type=str, default="data/ovbench.json", help="Path to the annotations")
parser.add_argument("--video_dir", type=str, default="", help="Root directory of source videos")
parser.add_argument("--result_dir", type=str, default="results", help="Root directory of results")
parser.add_argument("--mode", type=str, required=True, choices=["online", "offline"], help="Online of Offline model for testing")
parser.add_argument("--task", type=str, required=False, nargs="+", \
                    choices=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"], \
                    default=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"], \
                    help="Tasks to evaluate")
parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
parser.add_argument("--save_results", type=bool, default=True, help="Save results to a file")

parser.add_argument("--gpt_api", type=str, required=False, default="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")
parser.add_argument("--gemini_project", type=str, required=False, default="project-241105")
args = parser.parse_args()

if args.model == "GPT":
    model = GPT(args)
elif args.model == "Gemini":
    model = Gemini(args)

with open(args.anno_path, "r") as f:
    annotations = json.load(f)

for i, item in enumerate(annotations):
    annotations[i]["video"] = os.path.join(args.video_dir, item["video"])

backward_anno = []
realtime_anno = []
forward_anno = []
backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

for anno in annotations:
    if anno["task"] in args.task:
        if anno["task"] in backward_tasks:
            backward_anno.append(anno)
        elif anno["task"] in realtime_tasks:
            realtime_anno.append(anno)
        elif anno["task"] in forward_tasks:
            forward_anno.append(anno)

anno = {
    "backward": backward_anno,
    "realtime": realtime_anno,
    "forward": forward_anno
}

model.eval(anno, args.task, args.mode)