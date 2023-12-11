import requests
from bs4 import BeautifulSoup
import json
import os
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

path_to_extension = r'C:\coding\python\1.54.0_0'  # path to ublock extension

chrome_options = Options()
chrome_options.add_argument('load-extension=' + path_to_extension)


driver = webdriver.Chrome(options=chrome_options)
# time.sleep(10)


def get_danzi(page):
    url = 'https://zidian.18dao.net/liebiao' + '?page=' + str(page-1)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    danzis = soup.find(
        'div', class_='view-content').find('tbody').find_all('tr')

    danzi_list = []
    for danzi in danzis:
        danzi_font = danzi.find(
            'td', class_='views-field views-field-danzi').find('a').text
        danzi_zhuyin = danzi.find(
            'td', class_='views-field views-field-zhuyin').text.strip()
        danzi_zhuyin = list(danzi_zhuyin.split('\n'))
        danzi_list.append({
            'font': danzi_font,
            'zhuyin': danzi_zhuyin
        })

    return danzi_list


def main():
    total_list = []
    for i in tqdm(range(155, 216)):
        try:
            danzis = get_danzi(i)
            total_list += danzis
        except:
            print('error', i)
            continue
    with open('danzis2.json', 'w', encoding='utf-8') as f:
        json.dump(total_list, f, ensure_ascii=False, indent=4)


def merge():
    with open('danzis.json', 'r', encoding='utf-8') as f:
        danzis1 = json.load(f)
    with open('danzis2.json', 'r', encoding='utf-8') as f:
        danzis2 = json.load(f)
    danzis = danzis1 + danzis2
    with open('chinese_chars.json', 'w', encoding='utf-8') as f:
        json.dump(danzis, f, ensure_ascii=False, indent=4)


def get_emoji(character):
    url = f'https://zidian.18dao.net/danzi/{character}'
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    field = soup.find('fieldset')
    tests = field.find_all('a', title=True)
    for t in tests:
        if 'Emoji' in t['title']:
            return t.text
    return None


def char_to_emoji():
    with open('chinese_chars.json', 'r', encoding='utf-8') as f:
        chars = json.load(f)
    print(len(chars))
    with open('start_pos') as f:
        start_pos = int(f.read())
    for i, char in enumerate(chars):
        if i < start_pos:
            continue
        emoji = None
        try:
            start_pos = i
            emoji = get_emoji(char['font'])
            if emoji:
                char['emoji'] = emoji
                with open('chinese_chars.json', 'w', encoding='utf-8') as f:
                    json.dump(chars, f, ensure_ascii=False, indent=4)
                with open('start_pos', 'w') as f:
                    f.write(str(start_pos+1))
        except:
            print('error', char['font'])
        time.sleep(2)


if __name__ == '__main__':
    char_to_emoji()
