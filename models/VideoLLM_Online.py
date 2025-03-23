"""
videollm-online Eval Code

Weight from: 
- https://huggingface.co/chenjoya/videollm-online-8b-v1plus

Inference Code from:
- https://github.com/showlab/videollm-online/blob/main/demo/cli.py
- https://github.com/THUNLP-MT/StreamingBench/blob/main/src/model/VideollmOnline.py
"""

import os
import transformers
import subprocess
logger = transformers.logging.get_logger('liveinfer')
from moviepy.editor import VideoFileClip
from videollm_online.demo.inference import LiveInfer

from utils.OVOBench import OVOBenchOffline

def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        './ffmpeg/ffmpeg',
        '-y',
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '10',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]
    subprocess.run(command, check=True)

class EvalVideollmOnline(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self._model_init()

    def _model_init(self):
        self.liveinfer = LiveInfer()

    def inference(self, video_file_name, prompt):
        file = video_file_name
        inp = prompt
        duration = VideoFileClip(video_file_name).duration
        timestamp = duration

        self.liveinfer.reset()
        name, ext = os.path.splitext(file)
        name = name.split('/')[-1]
        ffmpeg_video_path = os.path.join('./cache', name + f'_{self.liveinfer.frame_fps}fps_{self.liveinfer.frame_resolution}' + ext)
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(file, ffmpeg_video_path, fps=self.liveinfer.frame_fps, resolution=self.liveinfer.frame_resolution)
        logger.warning(f'{file} -> {ffmpeg_video_path}, {self.liveinfer.frame_fps} FPS, {self.liveinfer.frame_resolution} Resolution')

        self.liveinfer.load_video(ffmpeg_video_path)
        self.liveinfer.input_query_stream(inp, video_time=timestamp)

        for i in range(self.liveinfer.num_video_frames):
            self.liveinfer.input_video_stream(i / self.liveinfer.frame_fps)
            query, response = self.liveinfer()

            if response:
                print(response)
                return response
        return None