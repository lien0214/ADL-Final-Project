import torch
from transformers import MT5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def calculate_perplexity(model, tokenizer, dataset, model_type='mt5', device='cuda'):
    model.to(device)
    model.eval()

    total_loss = 0
    total_length = 0

    for item in dataset:
        if model_type == 'mt5':
            input_ids = tokenizer.encode(item['input_text'], return_tensors="pt", add_special_tokens=True, max_length=512).to(device)
            target_ids = tokenizer.encode(item['target_text'], return_tensors="pt", add_special_tokens=True, max_length=512).to(device)
            decoder_input_ids = torch.cat([torch.tensor([[tokenizer.pad_token_id]]).to(device), target_ids[:, :-1]], dim=-1)
            outputs = model(input_ids=input_ids, labels=target_ids, decoder_input_ids=decoder_input_ids)
        elif model_type == 'llama':
            # Assuming the dataset format for Llama is slightly different
            input_ids = tokenizer.encode(item['prompt'], return_tensors="pt", add_special_tokens=True, max_length=512).to(device)
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

# Example usage
# Load your models and tokenizers here for mT5 and LoRA Llama2
# mT5_model, llama_model, mT5_tokenizer, llama_tokenizer = ...

# Load or prepare your dataset here
# dataset = load_dataset(...)  # Adjust according to your data format

# Calculate perplexity for mT5
# mT5_perplexity = calculate_perplexity(mT5_model, mT5_tokenizer, dataset, model_type='mt5')
# print("mT5 Perplexity:", mT5_perplexity)

# Calculate perplexity for LoRA Llama2
# llama_perplexity = calculate_perplexity(llama_model, llama_tokenizer, dataset, model_type='llama')
# print("LoRA Llama2 Perplexity:", llama
