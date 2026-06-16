# DentVLM batch inference.
#
# This script loads a trained DentVLM checkpoint, runs prediction on a JSON test
# set, and writes one JSON object per line for downstream evaluation.


import os
import re
import math
import json
import argparse
import itertools
from tqdm import tqdm
from qwen_vl_utils import fetch_image
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm import LLM, SamplingParams

def set_prompt(question: str):
    return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n")
def get_message(data_item, MIN_PIXELS, MAX_PIXELS):
    message_item = {
        "id": data_item['id'],
        "prompt": set_prompt(data_item['question']),
        "multi_modal_data": {
            "image": fetch_image({
                "image": data_item['images'][0], 
                "min_pixels": MIN_PIXELS, 
                "max_pixels": MAX_PIXELS
            })
        },
    }
    return message_item


def batched_iterable(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch


def main(args):
    with open(args.test_file, "r", encoding='utf-8') as F:
        test_data = json.load(F)
    unit = int(math.ceil(len(test_data)/args.proc_total))
    print(args.proc_id, len(test_data), unit*args.proc_id, min(unit*(args.proc_id+1), len(test_data)))
    test_data = test_data[unit*args.proc_id: min(unit*(args.proc_id+1), len(test_data))]

    if os.path.isfile(f"{args.output_path[:-5]}{args.proc_id}.json"):
        with open(f"{args.output_path[:-5]}{args.proc_id}.json", "r", encoding='utf-8') as F:
            cur_test_length = sum(1 for _ in F)
        print(cur_test_length)
        test_data = test_data[cur_test_length:]
    else:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
    id_info_mapping = {}
    for item in test_data:
        id_info_mapping[item['id']] = {
            "id": item['id'],
            "question": item['question'],
            "gt_ans": item['gt_ans'],
            "images": item['images'],
        }
        if 'area' in item:
            id_info_mapping[item['id']]['area'] = item['area']

    llm = LLM(
        model=args.model_dir,
        max_model_len=16384,
        max_num_seqs=args.batch_size,
        tensor_parallel_size=args.parallel_size,
        gpu_memory_utilization=0.8,
        worker_use_ray=True,
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=512,
        stop_token_ids=[],
    )

    for batch in tqdm(batched_iterable(test_data, args.batch_size), total=int(math.ceil(len(test_data)/args.batch_size)), desc=f"{args.proc_id}_batch_infer"):        
        messages = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(get_message, item, args.min_pixels, args.max_pixels): item for item in batch}
            for future in as_completed(futures):
                messages.append(future.result())

        responses = llm.generate(messages, sampling_params=sampling_params, use_tqdm=False)
        for msg, res in zip(messages, responses):
            generated_text = res.outputs[0].text
            # print(f"{msg['id']}\t{generated_text}")
            cur_item = {
                "id": msg['id'],
                "question": id_info_mapping[msg['id']]['question'],
                "pred_response": generated_text,
                "gt_ans": id_info_mapping[msg['id']]['gt_ans'],
                "images": id_info_mapping[msg['id']]['images'],
            }
            if 'area' in id_info_mapping[msg['id']]:
                cur_item['area'] = id_info_mapping[msg['id']]['area']
            with open(f"{args.output_path[:-5]}{args.proc_id}.json", "a", encoding='utf-8') as F:
                F.write(f"{json.dumps(cur_item, ensure_ascii=False)}\n")
                

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process test data and generate responses.")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the test data file in JSON format.")
    parser.add_argument('--output_path', type=str, required=True, help="Path prefix for output answer files.")
    parser.add_argument('--proc_total', type=int, required=True, help="Total number of parallel processes.")
    parser.add_argument('--proc_id', type=int, required=True, help="Index of the current process.")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--parallel_size', type=int, default=1)
    parser.add_argument('--min_pixels', type=int, default=4 * 28 * 28)
    parser.add_argument('--max_pixels', type=int, default=8192 * 28 * 28)

    args = parser.parse_args()

    # Run main function with provided arguments
    main(args)
