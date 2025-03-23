import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os
import base64
from utils.OVOBench import OVOBenchOffline

class EvalGemini(OVOBenchOffline):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.project = args.gemini_project
        self._init_model()

    def _init_model(self):
        self.proxy_on()
        vertexai.init(project=self.project, location="us-central1")
        self.vision_model = GenerativeModel(model_name="gemini-1.5-pro")

    def proxy_on(self):
        os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        print(os.environ['http_proxy'])

    def video_to_base64(self, video_path):
        # 读取视频文件的二进制数据
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
        
        # 将二进制数据编码为 Base64
        base64_encoded = base64.b64encode(video_data)
        
        # 将 Base64 编码的数据转换为字符串
        base64_string = base64_encoded.decode('utf-8')
        
        return base64_string

    def inference(self, video_file_name, prompt):
        video_file = self.video_to_base64(video_file_name)
        
        try:
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
            return response.text
        except Exception as e:
            print(e)
            return None