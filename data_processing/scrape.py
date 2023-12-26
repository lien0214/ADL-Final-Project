# this script is used to scrape the emoji data from https://www.emojiall.com/zh-hant/all-emojis

import requests
from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm

def get_emojis():
    url = 'https://www.emojiall.com/zh-hant/all-emojis'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    emojis = soup.find_all('div', class_='emoji_card')

    emoji_list = []
    for emoji in emojis:
        emoji_font = emoji.find('a', class_='emoji_font').text
        emoji_img = emoji.find('a',class_='emoji_name truncate').text
        emoji_list.append({
            'emoji': emoji_font,
            'meaning': emoji_img
        })

    with open('emoji.json', 'w', encoding='utf-8') as f:
        json.dump(emoji_list, f, ensure_ascii=False, indent=4)
        
    print(f"{len(emoji_list)} emojis scraped")

def get_emoji_related_words(emoji):
    url = f'https://www.emojiall.com/zh-hant/tag-cloud-page/{emoji}'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find_all('div', class_='emoji_card_list pages')[3].find('table')
    tags = table.find_all('tr')[1:]
    
    related_words = []
    for tag in tags:
        text = tag.find('a').text
        score = tag.find('p').text
        related_words.append({
            'tag': text,
            'score': score
        })
    return related_words
    
    
def main():
    with open('emoji.json', 'r', encoding='utf-8') as f:
        emojis = json.load(f)
    
    error_emojis = []
    for emoji in tqdm(emojis):
        try:
            emoji['related_words'] = get_emoji_related_words(emoji['emoji'])
        except:
            error_emojis.append(emoji['emoji'])
    
    with open('emoji.json', 'w', encoding='utf-8') as f:
        json.dump(emojis, f, ensure_ascii=False, indent=4)
    
    print(f"{len(error_emojis)} emojis failed to scrape.")
    print(error_emojis)
        
if __name__ == '__main__':
    main()