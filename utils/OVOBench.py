import abc
from tqdm import tqdm
import json
import os
import sys
sys.path.append("..")
from constant import BR_PROMPT_TEMPLATE, REC_PROMPT_TEMPLATE, SSR_PROMPT_TEMPLATE, CRR_PROMPT_TEMPLATE

class OVOBenchOnline():
    def __init__(self) -> None:
        pass

    def inference():
        pass

class OVOBenchOffline():
    def __init__(self, args):
        self.args = args

    def eval(self, anno, task_list, mode = "offline"):
        # Inference
        if len(anno["backward"]) > 0:
            backward_results = []
            for _anno_ in tqdm(anno["backward"], desc="Backward Tasks"):
                id = _anno_["id"]
                video = _anno_["video"]
                task = _anno_["task"]
                question = _anno_["question"]
                options = _anno_["options"]
                realtime = _anno_["realtime"]
                assert not question == None
                assert not options == None
                prompt = self.build_prompt(task = task, question = question, options = options, _anno_ = None, index = None)

                chunk_video_path = os.path.join(self.args.chunked_dir, f"{id}.mp4")

                assert os.path.exists(chunk_video_path)
                try:
                    response = self.inference(chunk_video_path, prompt)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    response = None

                result = {
                    "id": id,
                    "video": video,
                    "task": task,
                    "question": question,
                    "response": response,
                    "ground_truth": chr(65 + _anno_["gt"])
                }
                backward_results.append(result)

        if len(anno["realtime"]) > 0:
            realtime_results = []
            for _anno_ in tqdm(anno["realtime"], desc="Realtime Tasks"):
                id = _anno_["id"]
                video = _anno_["video"]
                task = _anno_["task"]
                question = _anno_["question"]
                options = _anno_["options"]
                realtime = _anno_["realtime"]
                assert not question == None
                assert not options == None
                prompt = self.build_prompt(task = task, question = question, options = options, _anno_ = None, index = None)

                chunk_video_path = os.path.join(self.args.chunked_dir, f"{id}.mp4")
                assert os.path.exists(chunk_video_path)

                try:
                    response = self.inference(chunk_video_path, prompt)
                except Exception as e:
                    print(f"Error during inference: {e}")
                    response = None

                result = {
                    "id": id,
                    "video": video,
                    "task": task,
                    "question": question,
                    "response": response,
                    "ground_truth": chr(65 + _anno_["gt"])
                }
                realtime_results.append(result)

        if len(anno["forward"]) > 0:
            forward_results = []
            for _anno_ in tqdm(anno["forward"], desc="Forward Tasks"):
                id = _anno_["id"]
                video = _anno_["video"]
                task = _anno_["task"]
                test_info = _anno_["test_info"]
                for i in range(len(test_info)):
                    prompt = self.build_prompt(task = task, question = None, options = None, _anno_ = _anno_, index = i)
                    realtime = test_info[i]["realtime"]

                    chunk_video_path = os.path.join(self.args.chunked_dir, f"{id}_{i}.mp4")
                    assert os.path.exists(chunk_video_path)
                    try:
                        response = self.inference(chunk_video_path, prompt)
                    except Exception as e:
                        print(f"Error during inference: {e}")
                        response = None
                    
                    _anno_["test_info"][i]["response"] = response
                forward_results.append(_anno_)
        
        # Calculate Score
        if len(anno["backward"]) == 0:
            backward_results = []
        if len(anno["realtime"]) == 0:
            realtime_results = []
        if len(anno["forward"]) == 0:
            forward_results = []

        # Save Results
        if self.args.save_results:
            os.makedirs(f"{self.args.result_dir}/{self.args.model}", exist_ok=True)
            with open(f"{self.args.result_dir}/{self.args.model}/{self.args.model}_{'_'.join(task_list)}_{mode}_1.json", "w") as f:
                json.dump({
                    "backward": backward_results,
                    "realtime": realtime_results,
                    "forward": forward_results
                }, f, indent=4)

    def build_prompt(self, task, question, options, _anno_, index):
        if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
            formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
            prompt = BR_PROMPT_TEMPLATE.format(question, formatted_options)
            
        elif task == "REC":
            activity = _anno_["activity"]
            question = "How many times did they " + activity + "?"
            prompt = REC_PROMPT_TEMPLATE.format(question)
        elif task == "SSR":
            step = _anno_["test_info"][index]["step"]
            prompt = SSR_PROMPT_TEMPLATE.format(step)
        elif task == "CRR":
            question = _anno_["question"]
            prompt = CRR_PROMPT_TEMPLATE.format(question)
        return prompt

    @abc.abstractmethod
    def inference(self, video_file_name, prompt, start_time=0, end_time=0):
        pass