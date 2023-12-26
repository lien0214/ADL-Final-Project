import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, MT5ForConditionalGeneration
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse


def perplexity(model, tokenizer, dataset, model_type='mt5', translate_to = "emoji", device='cuda'):

    instruction = 'chinese'
    target = 'emoji'
    if translate_to == 'chinese':
        instruction = 'emoji'
        target = 'chinese'


    total_loss = 0
    total_length = 0


    for item in dataset:
        if model_type == 'mt5':
            input_ids = tokenizer.encode(item[instruction], return_tensors="pt", add_special_tokens=True, max_length=512).to(device)
            target_ids = tokenizer.encode(item[target], return_tensors="pt", add_special_tokens=True, max_length=512).to(device)
            decoder_input_ids = torch.cat([torch.tensor([[tokenizer.pad_token_id]]).to(device), target_ids[:, :-1]], dim=-1)
            outputs = model(input_ids=input_ids, labels=target_ids, decoder_input_ids=decoder_input_ids)
        elif model_type == 'llama':
            # Assuming the dataset format for Llama is slightly different
            input_ids = tokenizer.encode(get_prompt(item[instruction]), return_tensors="pt", add_special_tokens=True, max_length=512).to(device)
            outputs = model(input_ids=input_ids)
            target_ids = input_ids[:, 1:]  # Shift input_ids to the right for targets

        with torch.no_grad():
            if model_type == 'mt5':
                loss = outputs.loss
            elif model_type == 'llama':
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss += loss.item() * input_ids.size(1)
        total_length += input_ids.size(1)

    average_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="",
        help="llama or mt5"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.model_type == "llama":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        model = PeftModel.from_pretrained(model, args.peft_path)
    else:
        model = MT5ForConditionalGeneration.from_pretrained(args.base_model_path).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.to("cuda")

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    model.eval()
    ppl = perplexity(model, tokenizer, data, args.model_type)
    print("Mean perplexity:", ppl["mean_perplexity"])
