from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import json
import time

# To use GPU:
#    1. Install tensorflow-gpu (see Installation)
#    2. Set CUDA_VISIBLE_DEVICES environment variable, e.g. os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#    3. Set disable_cuda=False, e.g. ws = WS("./data", disable_cuda=False)
# To use CPU:
ws = WS("./data", disable_cuda=False)
pos = POS("./data", disable_cuda=False)

print('Loading sentences...')

with open('../data/new_all_sentences.json', encoding='utf-8') as f:
    sentence_list = json.load(f)

sentence_list = sentence_list

word_sentence_list = ws(
    sentence_list,
    # sentence_segmentation = True, # To consider delimiters
    # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
    # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
    # coerce_dictionary = dictionary2, # words in this dictionary are forced
)

pos_sentence_list = pos(word_sentence_list)

print('Done.')

def print_word_pos_sentence(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    for word, pos in zip(word_sentence, pos_sentence):
        print(f"{word}({pos})", end="\u3000")
    print()
    return
    
data = []
for i, sentence in enumerate(sentence_list):
    print()
    print(f"'{sentence}'")
    print_word_pos_sentence(word_sentence_list[i],  pos_sentence_list[i])
    data.append({
        'sentence': sentence,
        'word': word_sentence_list[i],
        'pos': pos_sentence_list[i]
    })
    
with open('../data/all_sentences_segmented.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
    