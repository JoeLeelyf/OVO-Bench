import abc
from tqdm import tqdm
import json
import os

class OVBenchOnline():
    def __init__(self) -> None:
        pass

    def inference():
        pass

class OVBenchOffline():
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
                try:
                    response = self.inference(video, prompt, start_time=0, end_time=realtime)
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
                try:
                    response = self.inference(video, prompt, start_time=0, end_time=realtime)
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
                    try:
                        response = self.inference(video, prompt, start_time=0, end_time=realtime)
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
            prompt = f"""
                Question: {question}
                Options:
                {formatted_options}
                Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
                Do not include any additional text or explanation in your response.
            """
        elif task == "REC":
            activity = _anno_["activity"]
            question = "How many times did they " + activity + "?"
            prompt = f""" 
                You're watching a video in which people may perform a certain type of action repetively. 
                The person performing this kind of action are referred to as 'they' in the following statement.
                You're task is to count how many times have different people in the video perform this kind of action in total.
                One complete motion counts as one. 
                Now, answer the following question: {question}
                Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
                Do not include any additional text or explanation in your response.
            """
        elif task == "SSR":
            step = _anno_["test_info"][index]["step"]
            prompt = f"""
                You're watching a tutorial video which contain a sequential of steps. 
                The following is one step from the whole procedures: 
                {step}
                Your task is to determine if the man or woman in the video is currently performing this step.
                Answer only with “Yes” or “No”.
                Do not include any additional text or explanation in your response.
            """

        elif task == "CRR":
            question = _anno_["question"]
            answer = _anno_["answer"]
            prompt = f"""
                You're responsible of answering questions based on the video content. 
                The following question are relevant to the latest frames, i.e. the end of the video.
                {question}
                Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
                Answer only with “Yes” or “No”.
                Do not include any additional text or explanation in your response.
            """
        return prompt

    @abc.abstractmethod
    def inference(self, video_file_name, prompt, start_time=0, end_time=0):
        pass