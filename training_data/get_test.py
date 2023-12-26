import json

with open('test.json', encoding='utf-8') as f:
    test = json.load(f)

with open('train_mix.json', encoding='utf-8') as f:
    train = json.load(f)

with open('../data/algorithm_dataset.json', encoding='utf-8') as f:
    algorithm = json.load(f)
for a in algorithm:
    a['type'] = "algorithm"
    
with open('../data/gpt_dataset.json', encoding='utf-8') as f:
    gpt = json.load(f)
for g in gpt:
    g['type'] = "gpt"

train_ids = [t['id'] for t in train]
test_ids = [t['id'] for t in test]

mix = [(0.2,"algo"), (0.8,"gpt")]
for m, name in mix:
    train_gpt_len = int(m * len(train_ids))
    train_gpt_ids = train_ids[:train_gpt_len]
    train_algo_ids = train_ids[train_gpt_len:]
    print(len(train_gpt_ids), len(train_algo_ids))
    
    new_train = []
    
    for i in train_gpt_ids:
        for g in gpt:
            if g['id'] == i:
                new_train.append(g)
                break
            
    for i in train_algo_ids:
        for a in algorithm:
            if a['id'] == i:
                new_train.append(a)
                break

    new_train.sort(key=lambda x: x['id'])
    with open(f'train_{name}.json', 'w', encoding='utf-8') as f:
        json.dump(new_train, f, ensure_ascii=False, indent=4)
        
new_test = []
for id in test_ids:
    t = {"id": id}
    for g in gpt:
        if g['id'] == id:
            t['chinese'] = g['chinese']
            t["emoji-gpt"] = g['emoji']
            break
    
    for a in algorithm:
        if a['id'] == id:
            t["emoji-algo"] = a['emoji']
            break

    new_test.append(t)

with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(new_test, f, ensure_ascii=False, indent=4)
    
    
    
    


    