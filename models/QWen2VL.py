import torch
from utils.OVOBench import OVOBenchOffline
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

class EvalQWen2VL(OVOBenchOffline):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args
        self._model_init_()

    def _model_init(self):
        model_path = self.args.model_path
        self.llm = LLM(
            model = model_path,
            dtype=torch.bfloat16,
            gpu_memory_utilization=0.7,
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=256,
            stop_token_ids=[],
        )

        self.processor = AutoProcessor.from_pretrained(model_path)

    def inference(self, video_file_name, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_file_name,
                        "nframes": 64,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        response = outputs[0].outputs[0].text
        return response