<h1 align="center">
  <font color=#0088cc>O</font><font color=#060270>V</font><font color=#0088cc>O</font>-Bench: How Far is Your Video-LLMs from Real-World <font color=#0088cc>O</font>nline <font color=#060270>V</font>ide<b style="color: #0088cc;">O</b> Understanding?
</h1>

<p align="center">
  <a href="https://huggingface.co/datasets/JoeLeelyf/OVO-Bench" style="margin-right: 10px;"> 
    <img src="https://img.shields.io/badge/arXiv-2412.****-b31b1b.svg?logo=arXiv">
  </a>
  <a href="https://huggingface.co/datasets/JoeLeelyf/OVO-Bench" style="margin-right: 10px;"> 
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-ffd21e">
  </a>
  <a href="https://huggingface.co/datasets/JoeLeelyf/OVO-Bench"> 
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Paper-ffd21e">
  </a>
</p>

## Introduction
### 🌟 Three distinct problem-solving modes
-  **Backward Tracing**: trace back to past events to answer the question.
-  **Real-Time Visual Perception**: understand and respond to events as they unfold at the current timestamp.
-  **Forward Active Responding**: delay the response until sufficient future information becomes available to answer the question accurately.

### 💫Chain-of-Time Thinking Process
OVO-Bench evaluates Video-LLMs' ability to find temporal visual clues from ongoing input, allowing models to wait for sufficient evidence before responding. We term this approach the Video Chain-of-Time thinking process, analogous to Chain-of-Thought reasoning in LLMs.
<p align="center">
  <img src="images/VideoCoT.png" alt="Distribution of questions and video in OVO-Bench." width="100%">
</p>


### Dataset Statistics
-  **644** videos
-  **3,100** Queries
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
bash scripts/inference_Gemini.sh
```
All inference results will be saved under `./results/[MODEL_NAME]`. Then run our scoring scripts:
```bash
bash scripts/score_Gemini.sh
```
Scores will show in cli:
```txt
Offline Model: Gemini
Evaluate Backward Tracing...
Task: HLD, Acc: 52.69
Task: ASI, Acc: 75.68
Task: EPM, Acc: 58.59
Backward Avg.: 62.32

Evaluate Real-time Visual Perception...
Task: STU, Acc: 54.49
Task: OJR, Acc: 67.39
Task: ATR, Acc: 80.17
Task: FPD, Acc: 68.32
Task: ACR, Acc: 66.97
Task: OCR, Acc: 87.25
Realtime Avg.: 70.77

Evaluate Forward Active Responding...
Task: REC, Acc: 35.53
Task: SSR, Acc: 74.24
Task: CRR, Acc: 61.67
Forward Avg.: 57.15

Total Avg.: 65.25
```
To evaluate your own models, inherit `OVOBenchOffline/Online` class in `./utils/OVOBench.py` and implement your own inference pipeline. Refer to our provided models under `./models` for further details.

## License
OVO-Bench is released under `CC BY-NC-SA 4.0` license. By downloading our dataset from our website or other sources, the user agrees to adhere to the terms of `CC BY-NC-SA 4.0` and licenses of the source datasets

## 🫥 Experimental Results

## 📍 Citing OVO-Bench
```bibtex
@article{
}
```