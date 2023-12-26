model="../llama/Taiwan-LLM-13B-v2.0-chat"
data="../data/preprocessed/mix_dataset_test.json"

python3 ppl.py \
  --base_model_path "$model" \
  --peft_path "$1" \
  --test_data_path "$data"
