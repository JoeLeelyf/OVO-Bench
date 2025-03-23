"""
    Calculate scores given inference results JSON
"""

import argparse
import os
import json
import os
from utils.OVOBenchScore import OVOBenchOfflineScore, OVOBenchOnlineScore

parser = argparse.ArgumentParser(description='Eval OVBench')
parser.add_argument("--result_dir", type=str, default="results", help="Root directory of results")
parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
parser.add_argument("--mode", type=str, required=True, choices=["online", "offline"], help="Online of Offline model for testing")
args = parser.parse_args()

assert os.path.exists(os.path.join(args.result_dir, args.model))

results_paths = os.listdir(os.path.join(args.result_dir, args.model))
results = {
    "backward": [],
    "realtime": [],
    "forward": []
}
for result_path in results_paths:
    with open(os.path.join(args.result_dir, args.model, result_path), "r") as f:
        result = json.load(f)
        results["backward"] += result["backward"]
        results["realtime"] += result["realtime"]
        results["forward"] += result["forward"]

if args.model in ["GPT", "Gemini", "InternVL2", "QWen2VL_7B", "QWen2VL_72B", "QWen2VL_7B_", "QWen2VL_72B_", "LongVU", "LLaVA_OneVision", "LLaVA_Video", "videollm_online", "FlashVStream", "MiniCPM_o"]:
    score_model = OVOBenchOfflineScore(args, results)
else:
    raise ValueError(f"Unsupported model: {args.model}. Please implement the model.")

score_model.score()