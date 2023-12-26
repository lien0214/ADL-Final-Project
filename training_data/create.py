import json
import random

TESTING_DATA_SIZE = 679

with open('../data/algorithm_dataset.json', encoding='utf-8') as f:
    algorithm = json.load(f)
for a in algorithm:
    a['type'] = "algorithm"
    
with open('../data/gpt_dataset.json', encoding='utf-8') as f:
    gpt = json.load(f)
for g in gpt:
    g['type'] = "gpt"

def create_mix_dataset(gpt_percentage):
    total = len(algorithm)
    gpt_percentage = 0.5
    gpt_len = gpt_percentage * total
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
    
    return dataset
    
mixed = [["algo", 0.2], ["mix", 0.5], ["gpt", 0.8]]

for m in mixed:
    dataset = create_mix_dataset(m[1])
    random.shuffle(dataset)
    train = dataset[:-TESTING_DATA_SIZE]
    test = dataset[-TESTING_DATA_SIZE:]
    
    with open(f'train_{m[0]}.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=4)
    
    with open(f'test_{m[0]}.json', 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=4)
    
    print(f"Created {m[0]} dataset with {len(train)} training data and {len(test)} testing data")