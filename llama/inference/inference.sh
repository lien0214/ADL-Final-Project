model=""
data=""
dir=""
if [[ $1 == "llama" ]]; then
    model="../llama/Taiwan-LLM-13B-v2.0-chat"
    data="../data/preprocessed/traditional/gpt_alpaca_test.json"
    dir="../llama"
    python3 inference.py \
        --base_model_path "$model" \
        --peft_path "$2" \
        --test_data_path "$data" \
        --output_path "${dir}/prediction.json"
elif [[ $1 == "mt5" ]]; then
    model="../text_to_emoji/output/checkpoint-1500"
    data="../text_to_emoji/test.json"
    dir="../mt5"
    python3 inference_mt5.py \
        --base_model_path "$model" \
        --test_data_path "$data" \
        --output_path "${dir}/prediction.json"
fi
