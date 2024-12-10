import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
from utils.split_frames import split_videos, encode_video
from OVBench import OVBenchOffline

class Gemini(OVBenchOffline):
    def __init__(self, model, args):
        super().__init__(args)
        self.model = model
        self.args = args
        self.project = args.gemini_project
        self._init_model()

    def _init_model(self):
        self.proxy_on()
        vertexai.init(project=self.project, location="us-central1")
        self.vision_model = GenerativeModel(model_name=self.model)

    def proxy_on(self):
        os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        print(os.environ['http_proxy'])

    def inference(self, video_file_name, prompt, start_time=0, end_time=0):
        temp_file = split_videos(video_file_name, end_time, start_time)
        video_file = encode_video(temp_file)

        response = self.vision_model.generate_content(
            [
                Part.from_data(
                    data=video_file, mime_type="video/mp4"
                ),
                prompt,
            ],
            generation_config={
                "temperature": 0
            }
        )
        
        os.remove(temp_file)
        return response.text