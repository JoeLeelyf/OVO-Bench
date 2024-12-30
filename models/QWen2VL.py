import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from utils.split_frames import split_save_frames
from utils.OVBench import OVBenchOffline
from qwen_vl_utils import process_vision_info

class EvalQWen2VL(OVBenchOffline):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args
        self._model_init_()
    
    def _model_init_(self):
        path = self.args.model_path
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            path, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(path)

    def inference(self, video_file_name, prompt, start_time=0, end_time=0):
        frames_path = split_save_frames(video_file_name, start_time=start_time, end_time=end_time, max_frames=64)
        frames_path = ["file:///" + frame_path for frame_path in frames_path]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames_path,
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
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