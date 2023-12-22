from transformers import BitsAndBytesConfig
import torch


def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    # zero shot
    return f"你要回答用戶的問題。\n用戶: {instruction}\n助手:"
    # few shot
    # return "以下是幾個例子。\n" + \
    # "用戶一: 議雖不從，天下鹹重其言。\n翻譯成白話文：\n助手: 他的建議雖然不被采納，但天下都很敬重他的話。\n" + \
    # "用戶二: 他請求退休，但下詔不許。\n翻譯成文言文：\n助手: 求緻仕，詔不許。\n" + \
    # "用戶三: 高祖初，為內秘書侍禦中散。\n翻譯成現代文：\n助手: 高祖初年，任內秘書侍禦中散。\n" + \
    # "你要回答用戶的問題。\n" + \
    # f"用戶: {instruction}\n助手:"

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    return bnb_config
