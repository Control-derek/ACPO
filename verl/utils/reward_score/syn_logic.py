import re
import sys
import random
from .SynLogic.task2verifier import verifier_classes
import json


class Data:
    """
    Data class for game/corpus
    @param question: question of the game/corpus
    @param answer: answer of the game/corpus
    @param difficulty: difficulty of the game/corpus, from 1 to 10
    """
    def __init__(self, question: str, answer: str, difficulty: int = 1, metadata: dict = None, **kwargs):
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.metadata = metadata
        self.gpt_response = ""
        
    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response
        }
    
    def to_json_str(self):
        return json.dumps(self.to_json(), ensure_ascii=False)
    
    @classmethod
    def from_json_str(cls, json_str):
        json_data = json.loads(json_str)
        return cls(**json_data)
    
    @classmethod
    def from_json_dict(cls, json_dict):
        instance = cls(**json_dict)
        if 'gpt_response' in json_dict:
            instance.gpt_response = json_dict['gpt_response']
        return instance
    
    @classmethod
    def from_jsonl_file(cls, file_path):
        data_list = []
        with open(file_path, "r") as f:
            for line in f:
                json_data = json.loads(line)
                instance = cls(**json_data)
                if 'gpt_response' in json_data:
                    instance.gpt_response = json_data['gpt_response']
                data_list.append(instance)
        return data_list

def _extract_answer(solution_str):
    pattern = r'<answer>(.*?)</answer>'
    answers = re.findall(pattern, solution_str)
    if len(answers) == 0:
        raise ValueError("could not extract answer")
    return answers[0]

def compute_score(solution_str, ground_truth, extra_info=None):
    # use raw source here
    raw_data_source = extra_info["raw_data_source"]
    verifier = verifier_classes[raw_data_source]()
    game_data = Data.from_json_str(extra_info["game_data_str"])
    try:
        solution = _extract_answer(solution_str)
    except ValueError as e:
        score = 0.0
    else:
        score = float(verifier.verify(game_data, solution))

    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"game_data: {game_data}")
        print(f"solution_str: {solution_str}")
        print(f"score: {score}")
        print(f"--------------------------------")

    return score