import os
import json
import numpy as np
from metrics import *
import shutil
from tqdm import tqdm

dataset2metric = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score, "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score,
    "musique": qa_f1_score, "dureader": rouge_zh_score, "gov_report": rouge_score,
    "qmsum": rouge_score, "multi_news": rouge_score, "vcsum": rouge_zh_score,
    "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score, 
    "lsht": classification_score, "passage_retrieval_en": retrieval_score, "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score, "lcc": code_sim_score, "repobench-p": code_sim_score,
}

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in tqdm(zip(predictions, answers, lengths), total=len(predictions)):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in tqdm(zip(predictions, answers), total=len(predictions)):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results", exist_ok=True)
    for base in ["pred_e", "pred"]:
        if not os.path.exists(base):
            continue
        results = {}
        for model in os.listdir(base):
            if "USH" in model:
                continue
            results[model] = {}
            for dataset in os.listdir(f"{base}/{model}"):
                results[model][dataset] = {}
                all_files = [f for f in os.listdir(f"{base}/{model}/{dataset}") if f.endswith('.jsonl')]
                for compression_type in all_files:
                    print(f"Processing {base}/{model}/{dataset}/{compression_type}")
                    predictions, answers, lengths = [], [], []
                    with open(f"{base}/{model}/{dataset}/{compression_type}", "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            all_classes = data["all_classes"]
                            if "length" in data:
                                lengths.append(data["length"])
                    if base == "pred_e":
                        score = scorer_e(dataset, predictions, answers, lengths, all_classes)
                    else:
                        score = scorer(dataset, predictions, answers, all_classes)
                    results[model][dataset][compression_type.split(".")[-2]] = score
        with open(f"results/{base}.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)