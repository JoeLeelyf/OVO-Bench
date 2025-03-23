"""
Flash-VStream Eval Code

Weight from: 
- https://huggingface.co/openbmb/MiniCPM-o-2_6

Inference Code from:
- https://github.com/OpenBMB/MiniCPM-o?tab=readme-ov-file#multimodal-live-streaming

Inference Platform:
- 1*A100 80GB
"""
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer

from utils.OVOBench import OVOBenchOffline

def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)

    num_units = math.ceil(video.duration)
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        if flatten:
            contents.extend(["<unit>", image])
        else:
            contents.append(["<unit>", image])
    
    return contents

class EvalMiniCPM(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self._model_init()

    def _model_init(self):
        self.model = AutoModel.from_pretrained(self.args.model_path, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16)
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)

        self.model.init_tts()

    def inference(self, video_file_name, prompt):
        video_path=video_file_name
        sys_msg = self.model.get_sys_prompt(mode='omni', language='en')
        # if use voice clone prompt, please set ref_audio
        # ref_audio_path = '/path/to/ref_audio'
        # ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        # sys_msg = model.get_sys_prompt(ref_audio=ref_audio, mode='omni', language='en')

        contents = get_video_chunk_content(video_path)
        msg = {"role":"user", "content": contents + [prompt]}
        msgs = [sys_msg, msg]

        # please set generate_audio=True and output_audio_path to save the tts result
        generate_audio = False

        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.5,
            max_new_tokens=4096 * 4,
            omni_input=False, # please set omni_input=True when omni inference
            use_tts_template=True,
            generate_audio=generate_audio,
            max_slice_nums=1,
            use_image_id=False,
            return_dict=True,
            max_inp_length=4096 * 4,
        )
        print(res.text)