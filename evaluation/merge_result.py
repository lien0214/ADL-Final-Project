import json
import numpy as np

tasks = ['emoji-gpt-to-chinese', 'emoji-algo-to-chinese', 'chinese-to-emoji']

def merge_result(results, r, weight):
    if weight == 0:
        for t in tasks:
            results[t] = r[t]
    for t in tasks:
        for m in ['mt5', 'llama']:
            for d in results[t][m]:
                results[t][m][d] = (results[t][m][d] * weight + r[t][m][d])/ (weight + 1)
    return results
    

def get_result(data):
    result = {}
    for task in tasks:
        all_r = [d for d in data[task]]
        allnum = [d["score"] for d in all_r]
        u = np.mean(allnum)
        a = np.std(allnum)
        model_scores = {
            "mt5": {},
            "llama": {}
        }
        for i in range(len(all_r)):
            score = (all_r[i]['score']-u)/a + 5
            data_type = f"{all_r[i]['data_type']}-{all_r[i]['batch']}"
            model = all_r[i]['model']
            if data_type not in model_scores[model]:
                model_scores[model][data_type] = [score]
            else:
                model_scores[model][data_type].append(score)
        for model in model_scores:
            for data_type in model_scores[model]:
                model_scores[model][data_type] = np.mean(model_scores[model][data_type])
        
        
        result[task] = model_scores
    return result

def main():
    with open('results.json', "r", encoding='utf-8') as f:
        results = json.load(f)
    
    with open('hank huang.json', "r", encoding='utf-8') as f:
        data = json.load(f)
    
    if "people" in results:
        if data['name'] not in results['people']:
            results['people'].append(data['name'])
        else:
            print("already in")
            return
    else:
        results['people'] = [data['name']]
    
    results['people'] = list(set(results['people']))
    s = len(results['people']) - 1
    
    r = get_result(data)
    with open('r.json', "w", encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False, indent=4)
    
    results = merge_result(results, r, weight=s)
    with open('results.json', "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    main()