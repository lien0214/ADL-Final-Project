model=""
data=""
dir=""
if [[ $1 == "llama" ]]; then
    model="./llama/Taiwan-LLM-13B-v2.0-chat"
    data="./data/preprocessed/traditional/gpt_alpaca_test.json"
    dir="./llama"
elif [[ $1 == "mistral" ]]; then
    model="./mistral/Mistral-7B-v0.1-chinese"
    data="./data/preprocessed/simplified/gpt_alpaca_test.json"
    dir="./mistral"
fi

python3 inference.py \
  --base_model_path "$model" \
  --peft_path "$2" \
  --test_data_path "$data" \
  --output_path "${dir}/prediction.json"