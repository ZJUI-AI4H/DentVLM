# DentVLM: A Multimodal Vision Language Model for Comprehensive Dental Diagnosis and Enhanced Clinical Practice

<p align="center">
  <a href="https://huggingface.co/ZJU-AI4H/DentVLM"><b>Model Checkpoint</b></a> |
  <a href="#quick-start"><b>Quick Start</b></a> |
  <a href="#training"><b>Training</b></a> |
  <a href="#evaluation"><b>Evaluation</b></a>
</p>

## News

- **2026.06**: We release the DentVLM codebase, example data format, training scripts, inference script, evaluation script, and model checkpoint information.

## Overview

DentVLM is a multimodal vision-language model designed for comprehensive dental image understanding and diagnosis-oriented question answering. This repository provides the code and example data format for reproducing the DentVLM training, inference, and evaluation pipeline.

DentVLM is released for research use. Model outputs should not be used as the sole basis for clinical diagnosis or treatment decisions.

## Main Features

- **Dental multimodal understanding**: Supports dental image-question inputs for malocclusion and dental disease recognition.
- **Two-stage supervised fine-tuning**: Provides stage-1 and stage-2 training configurations based on Qwen2-VL and LLaMA-Factory.
- **Batch inference**: Includes a vLLM-based inference script for JSON-formatted test sets.
- **Automatic evaluation**: Computes accuracy, hit rate, and region-level IoU for the released task format.

## Model Checkpoint

The DentVLM checkpoint is hosted on [Hugging Face](https://huggingface.co/ZJU-AI4H/DentVLM). To request access to the model weights, please contact Z.L. at zuozhuliu@intl.zju.edu.cn.

After downloading, place the checkpoint under `checkpoints/` or update `--model_dir` / `model_dir` to the local checkpoint path.

## Repository Structure

```text
├── answer/                             # optional generated answers and evaluation outputs
├── checkpoints/                        # optional local model checkpoints
├── data/                               # example source data
│   ├── imgs/                               # source images for training and testing
│   ├── dataset_info.json                   # dataset registration for LLaMA-Factory
│   ├── inst_data_1st_train.json            # example training data for stage 1
│   ├── inst_data_2nd_train.json            # example training data for stage 2
│   └── test_set.json                       # example test set
├── packages/                           # third-party packages used for training
├── tokenized_data/                     # optional tokenized data cache
├── train_script/                       # training and DeepSpeed configurations
│   ├── ds_z2_config.json
│   ├── ds_z3_config.json
│   ├── qwen2vl_full_sft_stage1.yaml
│   └── qwen2vl_full_sft_stage2.yaml
├── get_score.py                        # evaluation script
├── inference.py                        # batch inference script
├── requirement.txt                     # Python dependencies
└── run.sh                              # end-to-end pipeline script
```

## Installation

### Requirements

- Linux, Ubuntu 20.04 or later recommended
- NVIDIA GPU
  - At least 24 GB VRAM recommended for inference
  - 80 GB VRAM recommended for training
- CUDA 12.1+
- Python 3.10.16

### Environment Setup

```bash
conda create -n dentvlm python=3.10.16
conda activate dentvlm

cd packages/LLaMA-Factory
pip install -e ".[torch,metrics]"

cd ../..
pip install -r requirement.txt

# FlashAttention is recommended for training and optional for single-case inference.
# Adjust MAX_JOBS based on your machine.
MAX_JOBS=32 pip install flash-attn==2.6.3 --no-build-isolation
```

Depending on the network environment and system configuration, installation may take from tens of minutes to several hours.

## Data Format

The example data files under `data/` show the expected format for training and evaluation.

Training data follows the ShareGPT-style multimodal format used by LLaMA-Factory:

```json
{
  "id": "en_dis_front_1394_Wedge-shaped Defect",
  "images": ["data/imgs/front_1394.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>Based on the imaging, does the patient have a wedge-shaped defect?"},
    {"from": "gpt", "value": "No"}
  ]
}
```

Test data contains the question, ground-truth answer, image path, and optional region annotation:

```json
{
  "id": "zh_dis_panoramic_3329_Impacted Tooth",
  "question": "根据影像，判断患者是否存在阻生牙？",
  "gt_ans": "是",
  "images": ["data/imgs/panoramic_3329.jpg"],
  "area": "上牙列的右侧后牙区"
}
```

## Quick Start

The following example runs single-case inference with a local DentVLM checkpoint.

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_dir = "checkpoints/DentVLM"
img_path = "data/imgs/example.jpg"
question = "Please determine Angle's classification of the patient."

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_dir)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": question},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, _ = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(f"DentVLM's Response: {output_text[0]}")
```

Depending on the hardware, model loading may take from several seconds to several minutes. Inference usually takes approximately 1-5 seconds per image-question pair.

## Training

DentVLM uses two-stage supervised fine-tuning. Before training, check the configuration files in `train_script/` and update paths, batch sizes, DeepSpeed settings, and GPU-related parameters for your environment.

```bash
FORCE_TORCHRUN=1 llamafactory-cli train train_script/qwen2vl_full_sft_stage1.yaml
FORCE_TORCHRUN=1 llamafactory-cli train train_script/qwen2vl_full_sft_stage2.yaml
```

You can also run the end-to-end script:

```bash
bash run.sh
```

## Batch Inference

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_dir checkpoints/DentVLM_2nd_train \
    --test_file data/test_set.json \
    --output_path answer/answer.json \
    --proc_total 1 \
    --proc_id 0
```

The script writes JSONL outputs such as `answer/answer0.json`. For multi-process inference, set `--proc_total` to the total number of processes and launch each process with a different `--proc_id`.

## Evaluation

```bash
python get_score.py \
    --input_path answer/answer.json \
    --output_path answer/answer_score.json \
    --proc_total 1
```

The evaluation script reports task-level scores and writes Excel summaries under `answer/`.

## Acknowledgement

This repository builds on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL), and [vLLM](https://github.com/vllm-project/vllm). We thank the authors and contributors of these projects.

## Contact

For technical questions about this repository, please contact Z.L. at zuozhuliu@intl.zju.edu.cn.

## Authors

Zijie Meng<sup>1,2</sup>, Jin Hao<sup>3</sup>, Xiwei Dai<sup>1,2</sup>, Yang Feng<sup>4</sup>, Jiaxiang Liu<sup>2</sup>, Bin Feng<sup>1</sup>, Huikai Wu<sup>4</sup>, Xiaotang Gai<sup>1,2</sup>, Hengchuan Zhu<sup>1,2</sup>, Tianxiang Hu<sup>1,2</sup>, Yangyang Wu<sup>2</sup>, Hongxia Xu<sup>5</sup>, Jin Li<sup>6</sup>, Jun Xiao<sup>2</sup>, Xiaoqiang Liu<sup>7</sup>, Joey Tianyi Zhou<sup>8</sup>, Fudong Zhu<sup>1</sup>, Zhihe Zhao<sup>9</sup>, Bing Fang<sup>3</sup>, Lunguo Xia<sup>3</sup>, Jimeng Sun<sup>10</sup>, Jian Wu<sup>2,5</sup>, Zuozhu Liu<sup>1,2,5</sup>

1. Stomatology Hospital, School of Stomatology, Zhejiang University School of Medicine, Zhejiang University, Hangzhou 310016, Zhejiang, China.  
2. College of Computer Science and Technology, Zhejiang University-University of Illinois Urbana-Champaign Institute, Zhejiang University, Hangzhou 310027, Zhejiang, China.  
3. Department of Orthodontics, Shanghai Ninth People's Hospital, College of Stomatology, Shanghai Jiao Tong University School of Medicine, Shanghai 200011, China.  
4. Angelalign Technology Inc., Shanghai 200082, China.  
5. Zhejiang Key Laboratory of Medical Imaging Artificial Intelligence, Haining 314400, Zhejiang, China.  
6. Department of Stomatology, The First Affiliated Hospital of Shenzhen University, Shenzhen Second People's Hospital, Shenzhen 518035, China.  
7. Department of Prosthodontics, Peking University School and Hospital of Stomatology, Beijing 100081, China.  
8. CFAR & IHPC, Agency for Science, Technology and Research, 138632, Singapore.  
9. State Key Laboratory of Oral Diseases, National Clinical Research Center for Oral Diseases, West China Hospital of Stomatology, Sichuan University, Chengdu, China.  
10. Siebel School of Computing and Data Science, University of Illinois Urbana-Champaign, Urbana, IL 61801, USA.
