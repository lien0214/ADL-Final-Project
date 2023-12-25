from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是一個表情符號翻譯助手，請協助將用戶的中文翻譯成表情符號。用戶:{instruction} 助手:"
    # 用戶:月圓花好。助手:🌕🌸 
def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    return bnb_config
