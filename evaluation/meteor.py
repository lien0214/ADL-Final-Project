import evaluate
import json
import argparse

meteor = evaluate.load('meteor')

def eval_chinese_to_emoji(data):
    predictions = []
    references_gpt = []
    references_algo = []
    for d in data:
        output = d['chinese_to_emoji'].replace(' ', '')
        emoji_gpt = d['emoji-gpt'].replace(' ', '')
        emoji_algo = d['emoji-algo'].replace(' ', '')
        predictions.append(output)
        references_gpt.append(emoji_gpt)
        references_algo.append(emoji_algo)
    scores_gpt = meteor.compute(predictions, references_gpt)
    scores_algo = meteor.compute(predictions, references_algo)
    together = [ a + b for a, b in zip(references_gpt, references_algo)]
    scores_together = meteor.compute(predictions, together)
    return {
        "gpt": sum(scores_gpt)/len(scores_gpt),
        "algo": sum(scores_algo)/len(scores_algo),
        "together": sum(scores_together)/len(scores_together)
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
    scores_gpt = meteor.compute(predictions_gpt, reference)
    scores_algo = meteor.compute(predictions_algo, reference)
    scores_together = [ (a + b)/2 for a, b in zip(scores_gpt, scores_algo)]
    return {
        "gpt": sum(scores_gpt)/len(scores_gpt),
        "algo": sum(scores_algo)/len(scores_algo),
        "together": sum(scores_together)/len(scores_together)
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
    elif "emoji-algo-to-chinese" in predictions[0] and "emoji-gpt-to-chinese" in predictions[0]:
        r = eval_emoji_to_chinese(predictions)
        result["emoji-to-chinese"] = r
    
    output_path = args.result_path
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
        
