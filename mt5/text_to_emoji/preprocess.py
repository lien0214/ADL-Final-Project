import json

with open('../data/mix_dataset.json', encoding='utf-8') as f:
    mix_dataset = json.load(f)

train = mix_dataset[:int(len(mix_dataset) * 0.8)]