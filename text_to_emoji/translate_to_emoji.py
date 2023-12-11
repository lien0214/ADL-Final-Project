import json

with open('../data/char_to_emoji.json', encoding='utf-8') as f:
    char_to_emoji = json.load(f)

with open('../data/zhuyin_to_char.json', encoding='utf-8') as f:
    zhuyin_to_chars = json.load(f) 

with open('../data/chinese_chars.json', encoding='utf-8') as f:
    chinese_chars = json.load(f) 

def translate_to_emoji(text):
    result = ''
    for char in text:
        if char in char_to_emoji:
            result += char_to_emoji[char]
        else:
            result += char
    return result

def main():
    with open('../data/all_sentences.json', encoding='utf-8') as f:
        all_sentences = json.load(f)
    
    result = []
    for s in all_sentences[:5]:
        emoji_translation = translate_to_emoji(s)
        result.append({
            'chinese': s,
            'emoji': emoji_translation
        })
    
    print(result)
    with open('../data/emoji_translation.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        
if __name__ == '__main__':
    main()
