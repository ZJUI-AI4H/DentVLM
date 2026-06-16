#!/bin/bash
# DentVLM training, inference, and evaluation pipeline.
#
# Review and update the paths and hardware-related settings in train_script/*.yaml
# before running this script on a new machine.


# Conduct two-stage training.
FORCE_TORCHRUN=1 llamafactory-cli train train_script/qwen2vl_full_sft_stage1.yaml
FORCE_TORCHRUN=1 llamafactory-cli train train_script/qwen2vl_full_sft_stage2.yaml

# Conduct batch inference.
CUDA_VISIBLE_DEVICES=0 python inference.py \
                        --model_dir checkpoints/DentVLM_2nd_train \
                        --test_file data/test_set.json \
                        --output_path answer/answer.json \
                        --proc_total 1 \
                        --proc_id 0

# Evaluate the results
python get_score.py \
    --input_path answer/answer.json \
    --output_path answer/answer_score.json \
    --proc_total 1
