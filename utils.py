from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"USER: {instruction} ASSISTANT:"
    # return f"### 任务:{instruction}\n### 回复:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    return bnb_config
