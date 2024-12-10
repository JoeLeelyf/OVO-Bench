import abc
from tqdm import tqdm
import json

class OVBenchOnline():
    def __init__(self) -> None:
        pass

    def inference():
        pass

class OVBenchOffline():
    def __init__(self, args):
        self.args = args

    def eval(self, anno, task, mode = "offline"):
        # Inference
        if len(anno["backward"]) > 0:
            backward_results = []
            for _anno_ in tqdm(anno["backward"], desc="Backward Tasks"):
                id = _anno_["id"]
                video = _anno_["video"]
                question = _anno_["question"]
                options = _anno_["options"]
                realtime = _anno_["realtime"]
                prompt = self.build_prompt(_anno_["task"], question, options)
                response = self.inference(video, prompt, start_time=0, end_time=realtime)

                result = {
                    "id": id,
                    "video": video,
                    "task": _anno_["task"],
                    "question": question,
                    "response": response,
                    "ground_truth": _anno_["ground_truth"]
                }
                backward_results.append(result)
        if len(anno["realtime"]) > 0:
            realtime_results = []
            for _anno_ in tqdm(anno["realtime"], desc="Realtime Tasks"):
                question = _anno_["question"]
                options = _anno_["options"]
                realtime = _anno_["realtime"]
                prompt = self.build_prompt(_anno_["task"], question, options)
                response = self.inference(_anno_["video_file_name"], prompt, start_time=0, end_time=realtime)

                result = {
                    "task": _anno_["task"],
                    "question": question,
                    "response": response,
                    "ground_truth": _anno_["ground_truth"]
                }
                realtime_results.append(result)
        
        # Calculate Score
        # if len(anno["backward"]) > 0:
        #     self.calculate_score_backward_realtime(backward_results)
        # if len(anno["realtime"]) > 0:
        #     self.calculate_score_backward_realtime(realtime_results)

        # Save Results
        if self.args.save_results:
            with open(f"{self.args.result_dir}/{self.args.model}_{task}_{mode}.json", "w") as f:
                json.dump({
                    "backward": backward_results,
                    "realtime": realtime_results
                }, f, indent=4)

    def build_prompt(self, task, question, options):
        if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
            prompt = f"""
                Question: {question}
                Options:
                {options}
                Answer with the option's letter from the given choices directly.
            """
        elif task == "REC":
            pass
        elif task == "SSR":
            pass
        elif task == "CRR":
            pass
        return prompt

    def calculate_score_backward_realtime(self, results):
        def get_score(response, gt):
            return response == gt
        # Calculate Score for Every Result
        for result in results:
            result["correct"] = get_score(result["response"], result["ground_truth"])

    @abc.abstractmethod
    def inference(self, video_file_name, prompt, start_time=0, end_time=0):
        pass