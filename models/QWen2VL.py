"""
Qwen2VL Eval Code

Weight from: 
- https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
- https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct

Inference Code from:
- https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

Inference Platform:
- 7B: 4*A100 80GB
- 72B: 8*A100 80GB
"""
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.OVOBench import OVOBenchOffline
from decord import VideoReader

def get_max_frames(video_file_name, max_frames):
    video = VideoReader(video_file_name)
    return min(max_frames, len(video) - 2)

class EvalQWen2VL(OVOBenchOffline):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args
        self._model_init()

    def _model_init(self):
        model_path = self.args.model_path
        self.model =  Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto", 
            attn_implementation="flash_attention_2"
        )

        self.processor = AutoProcessor.from_pretrained(model_path)

    def inference(self, video_file_name, prompt):
        frames_num = get_max_frames(video_file_name, max_frames=64)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_file_name,
                        "max_pixels": 360 * 420,
                        "nframes": frames_num,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text