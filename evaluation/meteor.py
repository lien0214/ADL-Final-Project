import evaluate
import json
import argparse

meteor = evaluate.load('meteor')

def eval_chinese_to_emoji(data):
    predictions = []
    references_gpt = []
    references_algo = []
    for d in data:
        output = d['chinese-to-emoji'].replace(' ', '')
        emoji_gpt = d['emoji-gpt'].replace(' ', '')
        emoji_algo = d['emoji-algo'].replace(' ', '')
        predictions.append(output)
        references_gpt.append(emoji_gpt)
        references_algo.append(emoji_algo)
        # r = meteor.compute(predictions=[output], references=[emoji_algo])
        # print(r)
    scores_gpt = meteor.compute(predictions=predictions, references=references_gpt)
    scores_algo = meteor.compute(predictions=predictions,references=references_algo)
    together = [ a + b for a, b in zip(references_gpt, references_algo)]
    scores_together = meteor.compute(predictions=predictions, references=together)
    return {
        "gpt": scores_gpt['meteor'],
        "algo": scores_algo['meteor'],
        "together": scores_together['meteor']
    }
    

def eval_emoji_to_chinese(data):
    reference = []
    predictions_gpt = []
    predictions_algo = []
    
    for d in data:
        chinese = d['chinese'].replace(' ', '')
        gpt_chinese = d["emoji-gpt-to-chinese"].replace(' ', '')
        algo_chinese = d["emoji-algo-to-chinese"].replace(' ', '')
        reference.append(chinese)
        predictions_gpt.append(gpt_chinese)
        predictions_algo.append(algo_chinese)
    scores_gpt = meteor.compute(predictions=predictions_gpt, references=reference)
    scores_algo = meteor.compute(predictions=predictions_algo, references=reference)
    return {
        "gpt": scores_gpt['meteor'],
        "algo": scores_algo['meteor'],
        "together": (scores_gpt['meteor'] + scores_algo['meteor']) / 2
    }
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction", "-p",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--result_path", "-r",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.prediction, "r", encoding='utf-8') as f:
        predictions = json.load(f)
        
    predictions.sort(key=lambda x: x['id'])
    result = {}
    if "chinese-to-emoji" in predictions[0]:
        r = eval_chinese_to_emoji(predictions)
        result["chinese-to-emoji"] = r
    if "emoji-algo-to-chinese" in predictions[0] and "emoji-gpt-to-chinese" in predictions[0]:
        r = eval_emoji_to_chinese(predictions)
        result["emoji-to-chinese"] = r
    
    output_path = args.result_path
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
