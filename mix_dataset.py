"""
How to use:
    python mix_dataset.py --gpt_percentage p    
"""

import argparse
import json
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpt_percentage', type=float, default=0.3, help='percentage of gpt sentences (any value higher than 0.2 will use the whole gpt dataset)')
args = parser.parse_args()


with open('data/algorithm_dataset.json', encoding='utf-8') as f:
    algorithm = json.load(f)
with open('data/gpt_dataset.json', encoding='utf-8') as f:
    gpt = json.load(f)

total = len(algorithm)
gpt_len = args.gpt_percentage * total
gpt_len = min(int(gpt_len), len(gpt))
gpt_index = [g['id'] for g in gpt]
random.shuffle(gpt_index)
gpt_index = gpt_index[:gpt_len]

dataset = []
for i in range(total):
    if i in gpt_index:
        for g in gpt:
            if g['id'] == i:
                dataset.append(g)
                break
    else:
        dataset.append(algorithm[i])
        
with open('data/mix_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)