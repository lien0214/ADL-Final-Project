model=""
data="./data/preprocessed/gpt_alpaca.json"

if [[ $1 == "llama" ]]; then
    model="./llama/Taiwan-LLM-13B-v2.0-chat"
elif [[ $1 == "mistral" ]]; then
    model="./mistral/Mistral-7B-v0.1-chinese"
fi

python3 ppl.py \
  --base_model_path "$model" \
  --peft_path "$2" \
  --test_data_path "$data"
