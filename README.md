<h1 align="center">
  <b style="color: #0088cc;">OVO</b>-Bench: How Far is Your Video-LLMs from Real-World <b style="color: #0088cc;">O</b>nline <b style="color: #0088cc;">V</b>ide<b style="color: #0088cc;">O</b> Understanding?
</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/JoeLeelyf/OVBench" style="margin-right: 10px;"> 
    <img src="https://img.shields.io/badge/arXiv-2412.****-b31b1b.svg?logo=arXiv">
  </a>
  <a href="https://huggingface.co/datasets/JoeLeelyf/OVBench" style="margin-right: 10px;"> 
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-ffd21e">
  </a>
  <a href="https://huggingface.co/datasets/JoeLeelyf/OVBench"> 
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-ffd21e">
  </a>
</p>

## Introduction
### 🌟 Three distinct problem-solving modes
-  **Backward Tracing**: trace back to past events to answer the question.
-  **Real-Time Visual Perception**: understand and respond to events as they unfold at the current timestamp.
-  **Forward Active Responding**: delay the response until sufficient future information becomes available to answer the question accurately.

### 💫Chain-of-Time Thinking Process
OVBench evaluates Video-LLMs' ability to find temporal visual clues from ongoing input, allowing models to wait for sufficient evidence before responding. We term this approach the Video Chain-of-Time thinking process, analogous to Chain-of-Thought reasoning in LLMs.
<p align="center">
  <img src="images/VideoCoT.png" alt="Distribution of questions and video in OVO-Bench." width="100%">
</p>


### Dataset Statistics
-  **971** videos
-  **3,097** QA pairs
<p align="center">
  <img src="images/data_num.jpg" alt="Distribution of averaged query timestamps and
video duration (in seconds) in OVOBench. " width="50%">
</p>

-  **263.42s** Average query timestamp.

<p align="center">
  <img src="images/data_duration.jpg" alt="Distribution of questions and video in OVO-Bench." width="50%">
</p>



##  Dataset Examples
<p align="center">
  <img src="images/benchmark_examples_vertical_00.png" alt="Distribution of questions and video in OVO-Bench." width="50%">
</p>

## Evaluation Pipeline

### Requirements
Following modules are required for inference and scoring pipeline.
```txt
moviepy==1.0.3
numpy
pillow
tqdm
```
Or run `pip insall -r requirements` to install all required modules.

### Data Preparation
Download `videos` and `annotations` from our [huggingface-repo](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench), unzip all files and place them under `./data` directory. 

### Inference and Score
We divide our evaluation pipeline into two parts: `inference` and `score`. For our released models, run our provided scripts under `./scripts` directory. For example, for InternVL2, run:
```bash
bash scripts/inference_InternVL2.sh
```
All inference results will be saved under `./results/[MODEL_NAME]`. Then run our scoring scripts:
```bash
bash scripts/score_InternVL2.sh
```
Scores will show in cli:
```txt
Offline Model: InternVL2
Evaluate Backward Tracing...
Task: EPM, Acc: 45.12
Task: HLD, Acc: 35.03
Task: ASI, Acc: 56.76
Backward Avg.: 44.70

Evaluate Real-time Visual Perception...
Task: STU, Acc: 48.31
Task: OJR, Acc: 52.72
Task: ATR, Acc: 68.97
Task: FPD, Acc: 68.32
Task: ACR, Acc: 59.63
Task: OCR, Acc: 73.83
Realtime Avg.: 60.57

Evaluate Forward Active Responding...
Task: CRR, Acc: 51.25
Task: REC, Acc: 28.92
Task: SSR, Acc: 59.43
Forward Avg.: 44.13

Total Avg.: 48.69
```
To evaluate your own models, inherit `OVBenchOffline/Online` class in `./utils/OVBench.py` and implement your own inference pipeline. Refer to our provided models under `./models` for further details.

## License
OVO-Bench is released under `CC BY-NC-SA 4.0` license. By downloading our dataset from our website or other sources, the user agrees to adhere to the terms of `CC BY-NC-SA 4.0` and licenses of the source datasets

## 🫥 Experimental Results

## 📍 Citing OVBench
```bibtex
@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```