import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
import json
import argparse
from tqdm import tqdm


MAX_LENGTH = 512
TO_EMOJI = True

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model_path",
    type=str,
    default="../../ADL/final/train/15epochs-4batch-to-chinese/checkpoint-10000",
    help="Path to the base model."
)
parser.add_argument(
    "--test_data_path",
    type=str,
    default="test.json",
    help="Path to test data."
)
parser.add_argument(
    "--output_path",
    type=str,
    default="output.json",
    help="Path for output json."
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Ensure CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = MT5ForConditionalGeneration.from_pretrained(args.base_model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    model.eval()

    predictions = []

    for item in tqdm(data[:20]):
        inputs = tokenizer(item["emoji"], return_tensors="pt", max_length=MAX_LENGTH, padding="max_length", truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=2048, num_return_sequences=1)
        
        generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        predictions.append({
            "id": item["id"],
            "instruction": item["emoji"],
            "answer": item.get("chinese", ""),
            "output": generated_output
        })

    with open(args.output_path, "w", encoding='utf-8') as json_file:
        json.dump(predictions, json_file, indent=2, ensure_ascii=False)
