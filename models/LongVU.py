"""
LongVU Eval Code

Weight from: 
- https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B

Inference Code from:
- https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B

Inference Platform:
- 1*A100 80GB
"""

import numpy as np
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader

from utils.OVOBench import OVOBenchOffline

class EvalLongVU(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self._model_init()
    
    def _model_init(self):
        pretrained = self.args.model_path
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            pretrained, None, "cambrian_qwen",
        )
        self.model.eval()

    def inference(self, video_file_name, prompt):
        video_path = video_file_name
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
        video = np.stack(video)
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, self.image_processor, self.model.config)
        video = [item.unsqueeze(0) for item in video]

        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return pred
            