from openai import OpenAI
import os
import os
from PIL import Image
import time
from utils.split_frames import split_frames, encode_image
from OVBench import OVBenchOffline

base_url = "https://api.openai.com/v1"

class GPT(OVBenchOffline):
    def __init__(self, args, model="gpt-4o"):
        super().__init__(args)
        self._init_model()
        self.args = args
        self.model_name = model

        self.api_key = args.gpt_api
    
    def _init_model(self):
        self.proxy_on()
        self.client = OpenAI(base_url= base_url, api_key=self.api_key)

    def proxy_on(self):
        os.environ['http_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['https_proxy'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTP_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        os.environ['HTTPS_PROXY'] = 'http://closeai-proxy.pjlab.org.cn:23128/'
        print(os.environ['http_proxy'])

    def build_prompt(self, question, urls):
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
                result = self.client.chat.completions.create(
                    model=model_name,
                    messages=message,
                    max_tokens=128,
                    temperature=0
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
    
    def inference(self, video_file_name, prompt, end_time, start_time=0):
        urls = []
        frames = split_frames(video_file_name, end_time=end_time, start_time=start_time)
        for frame in frames:
            frame_image = Image.fromarray(frame)
            base64_image = encode_image(frame_image)
            urls.append(f"data:image/png;base64,{base64_image}")
        
        prompt = self.build_prompt(prompt, urls)
        response = self.call_gpt_eval(prompt, self.model_name)
        return response
