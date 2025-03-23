"""
Flash-VStream Eval Code

Weight from: 
- https://huggingface.co/IVGSZ/Flash-VStream-7b

Inference Code from:
- https://github.com/IVGSZ/Flash-VStream/blob/main/flash_vstream/serve/cli_video_stream.py
- https://github.com/THUNLP-MT/StreamingBench/blob/main/src/model/FlashVstream.py

Inference Platform:
- 1*A100 80GB
"""

import argparse
import requests
import logging
import torch
import numpy as np
import time
import os

from flash_vstream.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from flash_vstream.conversation import conv_templates, SeparatorStyle
from flash_vstream.model.builder import load_pretrained_model
from flash_vstream.utils import disable_torch_init
from flash_vstream.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from torch.multiprocessing import Process, Queue, Manager
from transformers import TextStreamer
from decord import VideoReader
from datetime import datetime
from PIL import Image
from io import BytesIO

from utils.OVOBench import OVOBenchOffline

def load_video(video_path):
    vr = VideoReader(video_path)
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

class EvalFlashVStream(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self._model_init()

    def _model_init(self):
        model_name = get_model_name_from_path(self.args.model_path)
        model_base = None
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.args.model_path, model_base, model_name, device="cuda", device_map="auto")
    
    def inference(self, video_file_name, prompt):
        try:
            video = load_video(video_file_name)
            video = self.image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

            qs = prompt
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.002,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
                
            input_token_len = input_ids.shape[1]
                
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        except Exception as e:
            print(e)
            outputs = None

        return outputs
