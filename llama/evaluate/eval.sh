model=""
data=""
if [[ $1 == "llama" ]]; then
    model="../llama/Taiwan-LLM-13B-v2.0-chat"
    data="../data/preprocessed/traditional/gpt_alpaca_test.json"
elif [[ $1 == "mistral" ]]; then
    model="./mistral/Mistral-7B-v0.1-chinese"
    data="./data/preprocessed/simplified/gpt_alpaca_test.json"
fi

python3 ppl.py \
  --base_model_path "$model" \
  --peft_path "$2" \
  --test_data_path "$data"
