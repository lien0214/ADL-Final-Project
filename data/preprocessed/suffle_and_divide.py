import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to input json file.')

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def shuffle_and_split(json_data, split_ratio=0.8):

    random.shuffle(json_data)
    for i, data in enumerate(json_data):
        data['id'] = i
    split_index = int(len(json_data) * split_ratio)
    train_data = json_data[:split_index]
    test_data = json_data[split_index:]

    return train_data, test_data

args = parser.parse_args()

input_file_path = args.input
output_train_path = args.input.replace('.json', '_train.json')
output_test_path = args.input.replace('.json', '_test.json')

json_data = read_json_file(input_file_path)

train_data, test_data = shuffle_and_split(json_data)

write_json_file(output_train_path, train_data)
write_json_file(output_test_path, test_data)