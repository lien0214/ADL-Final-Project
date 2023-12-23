import json
import random

with open('../data/char_to_emoji.json', encoding='utf-8') as f:
    char_to_emoji = json.load(f)
    
def translate_to_emoji(text):
    result = ''
    last = ""
    for char in text:
        if char not in char_to_emoji:
            result += char
            continue
        
        r = char_to_emoji[char]
        c = ""
        if r['emoji']:
            c = r['emoji'][0]
        else:
            c = random.choice(r['zhuyin_emoji'])
        if last != c:
            result += c
        last = c
            
    return result

def main():
    with open("../data/all_sentences.json", encoding='utf-8') as f:
        sentences = json.load(f)
    
    emoji_sentence = []
    for i,sentence in enumerate(sentences):
        emoji_sentence.append({
            'emoji': translate_to_emoji(sentence),
            'chinese': sentence,
            'id': i
        })
        
    with open('../data/algorithm_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(emoji_sentence, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    main()
