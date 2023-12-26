import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import json
from transformers import MT5ForConditionalGeneration
import argparse

MAX_LENGTH = 512

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the base model."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        required=True,
        help="Path for output json."
    )
    args = parser.parse_args()

    model = MT5ForConditionalGeneration.from_pretrained(args.base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    model.eval()

    data_size = int(len(data)/10)
    instructions = [x["chinese"] for x in data]
    ids = [x["id"] for x in data]  

    tokenized_instructions = tokenizer(instructions, max_length=MAX_LENGTH, padding="max_length", truncation=True)
    output_masks = []

    predictions = []

    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0).to("cuda")
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0).to("cuda")
        label = input_ids

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, max_new_tokens=2048, num_return_sequences=1)
        generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(instructions[i]):].strip()

        prediction = {
            "id": ids[i],
            "instruction": data[i]["chinese"],
            "answer": data[i]["emoji"],
            "output": generated_output
        }
        predictions.append(prediction)

    with open(args.output_path, "w", encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)
