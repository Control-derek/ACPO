import datasets
import os
import argparse
import json

base_prompt_template = """
A conversation between User and Assistant.
User: {}

Assistant: <think>
""".strip()

qwen_instruct_prompt_template = """
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n {}<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>
""".strip()

llama_instruct_prompt_template = """
<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 17 Oct 2025\n\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet me solve this step by step.\n<think>
""".strip()

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir', default='~/data/sync_logic')
parser.add_argument('--template_type', type=str, default='base')

args = parser.parse_args()

def make_prompt(dp, template_type):
    user_prompt = dp['prompt'][0]["content"]
    if template_type == 'base':
        return base_prompt_template.format(user_prompt)
    elif template_type == 'qwen-instruct':
        return qwen_instruct_prompt_template.format(user_prompt)
    elif template_type == "llama-instruct":
        return llama_instruct_prompt_template.format(user_prompt)
    else:
        raise NotImplementedError

def make_map_fn(split):
    def process_fn(example, idx):
        prompt = make_prompt(example, template_type=args.template_type)

        extra_info = example.pop("extra_info")
        game_data_str = extra_info["game_data_str"]
        if game_data_str is None:
            question = extra_info["original_question"]
            answer = extra_info["original_answer"]
            difficulty = 1
            metadata = {}
            game_data_str = json.dumps(
                {
                    "question": question,
                    "answer": answer, 
                    "difficulty": difficulty,
                    "metadata": metadata
                }
            )
        data = {
            "data_source": "syn_logic",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "style": "",
                "solution": "",
                "answer": "",
                "ground_truth": "",
            },
            "extra_info": {
                "raw_data_source": example["data_source"].replace("val/", ""),
                "split": split,
                "index": idx,
                "game_data_str": game_data_str,
            }
        }
        return data
    return process_fn



dataset = datasets.load_dataset("MiniMaxAI/SynLogic", 'easy')
train_dataset = dataset["train"]
test_dataset = dataset["validation"]

train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

local_dir = os.path.expanduser(args.local_dir)
train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))