import jsonlines
import json
import random

TRAIN_SIZE = 0.8

answer = ""
question = ""
system_prompt = "You are an expert in translating Chinese to Emojis."
s = {
    "messages": [
        {"role": "system", "content": f"{system_prompt}"}, 
        {"role": "user", "content": f"{question}"}, 
        {"role": "assistant", "content": f"{answer}"},
    ]
}

def main():
    
    with open('../data/mix_dataset.json', "r", encoding='utf-8') as f:
        data = json.load(f)
    
    preprocessed = []
    for d in data:
        question = d['chinese']
        answer = d['emoji']
        s = {
            "messages": [
                {"role": "system", "content": f"{system_prompt}"}, 
                {"role": "user", "content": f"{question}"}, 
                {"role": "assistant", "content": f"{answer}"},
            ]
        }
        preprocessed.append(s)
        
    random.shuffle(preprocessed)
    
    l = int(len(preprocessed) * TRAIN_SIZE)
        
    with open('train.jsonl', 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(preprocessed[:l])
        
    with open('test.jsonl', 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(preprocessed[l:])
        
if __name__ == "__main__":
    main()