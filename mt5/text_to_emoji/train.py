from transformers import Trainer
from transformers import MT5ForConditionalGeneration
from transformers import TrainingArguments
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import json
import wandb
import argparse

wandb.init(project='my_mt5_project')
parser = argparse.ArgumentParser()
parser.add_argument("--max_length", type=int, default=192)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--to_emoji", action='store_true')
args = parser.parse_args()

TRAINING_DATA_SIZE  = 4000
# MAX_LENGTH          = 128
# EPOCH               = 10
# BATCH_SIZE          = 4
# TO_EMOJI            = False
MAX_LENGTH = args.max_length
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
TO_EMOJI = args.to_emoji
TO_CHINESE          = not TO_EMOJI
output_dir = f"{EPOCH}epochs-{BATCH_SIZE}batch-{'to-emoji' if TO_EMOJI else 'to-chinese'}"

    
with open('../data/mix_dataset.json', encoding='utf-8') as f:
    mix_dataset = json.load(f)

train = mix_dataset[:TRAINING_DATA_SIZE]
test = mix_dataset[TRAINING_DATA_SIZE:]

with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(train, f, ensure_ascii=False, indent=4)

with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=4)

if TO_EMOJI:
    train_data = {
        'id': [t['id'] for t in train],
        'input': [t['chinese'] for t in train],
        'target': [t['emoji'] for t in train],
    }
else:
    train_data = {
        'id': [t['id'] for t in train],
        'input': [t['emoji'] for t in train],
        'target': [t['chinese'] for t in train],
    }

raw_datasets = Dataset.from_dict(train_data)
checkpoint = "google/mt5-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_function(examples):
    padding = "max_length"
    inputs = examples['input']
    targets = examples['target']
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=MAX_LENGTH, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    # labels["input_ids"] = [
    #     [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    # ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=False)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCH,
    per_device_train_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    report_to="wandb",  # Enable wandb logging
    run_name=output_dir  # Optional: Give a name to the run
)
model = MT5ForConditionalGeneration.from_pretrained(checkpoint)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(f"./models/{output_dir}")

wandb.finish()