import os
import json
import copy
import datasets


template = """
<|im_start|>system
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>
<|im_start|>user
Answer the following question:
{question}

Please first think step by step, then output your answer in the \\boxed{{}} at last.
<|im_end|>
<|im_start|>assistant
Let's think step by step and output the final answer within \\boxed{{}}.
"""

def format_prompt(question):
    return template.format(question=question)
    

local_dir = os.path.expanduser("~/data/open_reasoner_zero")
if os.path.exists(local_dir) is False:
    os.makedirs(local_dir)
train_data_path = os.path.expanduser("~/Open-Reasoner-Zero/data/orz_math_57k_collected.json")
eval_math_data_path = os.path.expanduser("~/Open-Reasoner-Zero/data/eval_data/math500.json")
eval_aime_data_path = os.path.expanduser("~/Open-Reasoner-Zero/data/eval_data/aime2024.json")

# None should be replaced
entry_template = {
    "data_source": "open_reasoner_zero",
    "prompt": [{
        "role": "user",
        "content": None,
    }],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": None
    },
    "extra_info": {
        'split': None,
        'index': None,
    }
}

with open(train_data_path) as f:
    train_data = json.load(f)
train_parsed_list = []
for data_i, one_data in enumerate(train_data):
    parsed_entry = copy.deepcopy(entry_template)
    parsed_entry["prompt"][0]["content"] = format_prompt(one_data[0]["value"])
    parsed_entry["reward_model"]["ground_truth"] = one_data[1]["ground_truth"]["value"]
    parsed_entry["extra_info"]["index"] = data_i
    parsed_entry["extra_info"]["split"] = "train"
    train_parsed_list.append(parsed_entry)

eval_parsed_list1 = []
with open(eval_math_data_path) as f:
    eval_data = json.load(f)
for data_i, one_data in enumerate(eval_data):
    assert len(one_data["prompt"]) == 1
    parsed_entry = copy.deepcopy(entry_template)
    parsed_entry["data_source"] = "math500"
    parsed_entry["prompt"][0]["content"] = format_prompt(one_data["prompt"][0]["value"])
    parsed_entry["reward_model"]["ground_truth"] = one_data["final_answer"]
    parsed_entry["extra_info"]["index"] = data_i
    parsed_entry["extra_info"]["split"] = "test"
    eval_parsed_list1.append(parsed_entry)

eval_parsed_list2 = []
with open(eval_aime_data_path) as f:
    eval_data = json.load(f)
for data_i, one_data in enumerate(eval_data):
    assert len(one_data["prompt"]) == 1
    parsed_entry = copy.deepcopy(entry_template)
    parsed_entry["data_source"] = "aime2024"
    parsed_entry["prompt"][0]["content"] = format_prompt(one_data["prompt"][0]["value"])
    parsed_entry["reward_model"]["ground_truth"] = one_data["final_answer"]
    parsed_entry["extra_info"]["index"] = data_i
    parsed_entry["extra_info"]["split"] = "test"
    eval_parsed_list2.append(parsed_entry)

train_dataset = datasets.Dataset.from_list(train_parsed_list)
test_dataset1 = datasets.Dataset.from_list(eval_parsed_list1)
test_dataset2 = datasets.Dataset.from_list(eval_parsed_list2)

train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
test_dataset1.to_parquet(os.path.join(local_dir, 'test1.parquet'))
test_dataset2.to_parquet(os.path.join(local_dir, 'test2.parquet'))

eval_amc_data_path = os.path.expanduser("~/understand-r1-zero/datasets/evaluation_suite/amc/")
eval_parsed_list_amc = []
eval_data = datasets.load_from_disk(eval_amc_data_path)
for data_i, one_data in enumerate(eval_data):
    parsed_entry = copy.deepcopy(entry_template)
    parsed_entry["data_source"] = "amc"
    parsed_entry["prompt"][0]["content"] = str(format_prompt(one_data["problem"]))
    parsed_entry["reward_model"]["ground_truth"] = str(one_data["answer"])
    parsed_entry["extra_info"]["index"] = data_i
    parsed_entry["extra_info"]["split"] = "test"
    eval_parsed_list_amc.append(parsed_entry)
test_dataset_amc = datasets.Dataset.from_list(eval_parsed_list_amc)
test_dataset_amc.to_parquet(os.path.join(local_dir, 'test_amc.parquet'))


print("be careful of question with multiple answers in minerva")
eval_minerva_data_path = os.path.expanduser("~/understand-r1-zero/datasets/evaluation_suite/minerva/")
eval_parsed_list_minerva = []
eval_data = datasets.load_from_disk(eval_minerva_data_path)
for data_i, one_data in enumerate(eval_data):
    answer = one_data["answer"][0].strip()
    parsed_entry = copy.deepcopy(entry_template)
    parsed_entry["data_source"] = "minerva"
    parsed_entry["prompt"][0]["content"] = str(format_prompt(one_data["problem"]))
    parsed_entry["reward_model"]["ground_truth"] = str(answer)
    parsed_entry["extra_info"]["index"] = data_i
    parsed_entry["extra_info"]["split"] = "test"
    eval_parsed_list_minerva.append(parsed_entry)
test_dataset_minerva = datasets.Dataset.from_list(eval_parsed_list_minerva)
test_dataset_minerva.to_parquet(os.path.join(local_dir, 'test_minerva.parquet'))


