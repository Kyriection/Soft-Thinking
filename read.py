
import os
import json
import numpy as np
import argparse
from get_math_results import main as eval_main
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="results/gsm")
parser.add_argument("--model", type=str, default=None)
args = parser.parse_args()

eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)

def calculate_token_cost(res_path, tokenizer):

    with open(res_path, "r") as f:
        lines = f.readlines()
        results = [json.loads(line) for line in lines]

    token_budget = []
    for output in results:
        token_cost = tokenizer.encode(output['model_generation'][0])
        token_budget.append(len(token_cost))
    print(f"Samples: {len(token_budget)} Total token cost: {np.mean(token_budget)}")

tokenizer = AutoTokenizer.from_pretrained(args.model)
calculate_token_cost(os.path.join(args.save_dir, "predictions.jsonl"), tokenizer)

