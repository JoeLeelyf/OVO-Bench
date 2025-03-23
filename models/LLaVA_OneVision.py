"""
LLaVA-OneVision Eval Code

Weight from: 
- https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov

Inference Code from:
- https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/playground/demo/video_demo.py

Inference Platform:
- 4*A100 80GB
"""

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import numpy as np
import copy
import warnings
from decord import VideoReader, cpu

warnings.filterwarnings("ignore")
device = "cuda"

from utils.OVOBench import OVOBenchOffline

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time

class EvalLLaVAOneVision(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self._model_init()
    
    def _model_init(self):
        pretrained = self.args.model_path
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)

    def inference(self, video_file_name, prompt):
        image_tensors = []
        image_sizes = []

        video,frame_time,video_time = load_video(video_file_name, 64, 1, force_sample=True)
        frames = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)
        image_sizes = [frame.size for frame in video]
        modality = "video"

        # Prepare conversation input
        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        # Generate response
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=[modality],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        response = text_outputs[0]
        return response
