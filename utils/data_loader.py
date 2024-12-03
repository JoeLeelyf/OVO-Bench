import json
import math
from tqdm import tqdm

def create_prompt(question, options, GT):
    pass

# def Model_Init():
#     pass

class Model_Init:

    def __init__(self):
        pass
    
    def inference(self, prompt, video_path, RealTime):
        pass

def load_datasets(eval_dataset_path):
    with open(eval_dataset_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    Questions = []
    Options = []
    Video_path = []
    RealTime = []
    GroundTruth = []
    for i , data in enumerate(datas):
        for j, q in enumerate(data["QA"]):
            Questions.append(q["Question"])
            Options.append(data["MC"][j]["Options"])
            Video_path.append(f"{data['folder']}/{data['video']}")
            RealTime.append(q["RealTime"])
            label_number = data["MC"][j]["label"]
            label_letter = chr(ord('A') + label_number)
            GroundTruth.append(label_letter)
    return Questions, Options, Video_path, RealTime, GroundTruth

def Evaluate():
    pass

def main():

    eval_dataset_path = "PATH_TO_EVAL_DATASET"
    Questions, Options, Video_path, RealTimes ,GroundTruth= load_datasets(eval_dataset_path)

    # Load the model
    model = Model_Init()

    # Assuming questions, video_paths, st_ed_times, and answers are defined lists
    for i, question in tqdm(enumerate(Questions), total=len(Questions), desc="Processing videos"):
        Question, Gt = create_prompt(question, Options[i], GroundTruth[i])
        video_path = Video_path[i]
        RealTime = math.ceil(RealTimes[i])
        prompt = Question

        # Inference
        response = model.inference(prompt, video_path, RealTime)

        # Evaluate the inference
        Evaluate(response, Gt)


if __name__ == '__main__':
    main()