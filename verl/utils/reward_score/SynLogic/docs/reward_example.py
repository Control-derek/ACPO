import time
import random


from pydantic import BaseModel
import time
import random
from task2verifier import *
from base.data import Data
import re



THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


def _extract_answer(text):
    # Define regex pattern to match content between <answer> and </answer>
    pattern = r'<answer>(.*?)</answer>'
    
    # Use re.search to find the first match
    match = re.search(pattern, text, re.DOTALL)
    
    # If match found, return the matched content
    if match:
        return match.group(1).strip()
    else:
        return None

def _extract_solution_with_thought(solution_str):
    model_output = solution_str
    
    if THOUGHT_DELIMITER_END in solution_str:
        model_output = solution_str.split(THOUGHT_DELIMITER_END)[1]
    
    predict_answer = _extract_answer(model_output)
    if predict_answer is not None:
        return predict_answer
    else:
        return model_output
    

class Payload(BaseModel):
    response: str
    answer: str
    prompt: str
    solution: str
    data_source: str
    game_data: str


def synthetic_puzzle_process(payload_dict):
    format_res = 0
    # check format
    if (payload_dict["response"].startswith('<think>') and
        payload_dict["response"].endswith('</answer>') and
        payload_dict["response"].count('<think>') == 1 and
        payload_dict["response"].count('</think>') == 1 and
        payload_dict["response"].count('<answer>') == 1 and
        payload_dict["response"].count('</answer>') == 1 ):
        format_res = 1
    logic_res = 0
    if format_res > 0:
        verifier_class = verifier_classes[payload_dict['data_source']]
        verifier = verifier_class()
        game_data = Data.from_json_str(payload_dict['game_data'])
        test_solution = payload_dict['response']
        test_solution = _extract_solution_with_thought(test_solution)
        logic_res = verifier.verify(game_data, test_solution)

    ## reward merge
    final_score = logic_res * format_res

    result = {
            "rewards": {
                "format_reward": format_res,
                "accuracy_reward": logic_res,
                "final_reward": final_score
            },
        }

    payload_dict['result'] = result

    return result
