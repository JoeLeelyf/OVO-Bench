from openai import OpenAI
import os
from PIL import Image
import time
from utils.OVOBench import OVOBenchOffline
import base64
import io
import numpy as np
from decord import cpu, VideoReader

base_url = "https://api.openai.com/v1"

class EvalGPT(OVOBenchOffline):
    def __init__(self, args, model="gpt-4o"):
        super().__init__(args)
        self.args = args
        self.model_name = model
        self.api_key = args.gpt_api
        print(self.api_key)
        
        self._init_model()
    
    def _init_model(self):
        self.proxy_on()
        self.client = OpenAI(base_url= base_url, api_key=self.api_key)

    def proxy_on(self):
        os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        print(os.environ['http_proxy'])

    def load_video(self, video_path, max_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = float(vr.get_avg_fps())
        
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
    
    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def build_messages(self, question, urls):
        message = []
        for url in urls:
            message.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": "low"
                    },
                }
            )
        message.append(
            {
                "type": "text",
                "text": question,
            }
        )

        prompt =  [
            {
                "role": "user",
                "content": message
            }
        ]
        return prompt
    
    def call_gpt_eval(self, message, model_name, retries=10, wait_time=1):
        for i in range(retries):
            try:
                result = self.client.beta.chat.completions.parse(
                    model=model_name,
                    messages=message,
                    max_tokens=128
                )
                response_message = result.choices[0].message.content 
                return response_message
            except Exception as e:
                if i < retries - 1:
                    print(f"Failed to call the API {i+1}/{retries}, will retry after {wait_time} seconds.")
                    print(e)
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Failed to call the API after {retries} attempts.")
                    print(e)
                    raise
    
    def inference(self, video_file_name, prompt):
        urls = []
        frames = self.load_video(video_path=video_file_name, max_frames_num=64)
        for frame in frames:
            frame_image = Image.fromarray(frame)
            base64_image = self.encode_image(frame_image)
            urls.append(f"data:image/png;base64,{base64_image}")
        
        prompt = self.build_messages(prompt, urls)
        response = self.call_gpt_eval(prompt, self.model_name)
        return response
