from opencc import OpenCC
import argparse
import json

toConvert = 'output'
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True, help='Path to input json file.')

args = parser.parse_args()

cc = OpenCC('tw2sp')

with open(args.input_path, 'r') as f:
    data = json.load(f)

for i, d in enumerate(data):
    data[i][toConvert] = cc.convert(d[toConvert])

with open('./simplified/'+args.input_path, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

