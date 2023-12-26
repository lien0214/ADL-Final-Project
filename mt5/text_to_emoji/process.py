import json
import time

with open('../data/conversations.json', encoding='utf-8') as f:
    conversations = json.load(f)
    
with open('../data/gpt_generate.json', encoding='utf-8') as f:
    gpt_generate = json.load(f)

all_sentences = []

for gpt in gpt_generate:
    all_sentences += [gpt['chinese']]
    
for conversation in conversations:
    all_sentences += conversation['chinese']
    
with open('../data/all_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(all_sentences, f, ensure_ascii=False, indent=4)