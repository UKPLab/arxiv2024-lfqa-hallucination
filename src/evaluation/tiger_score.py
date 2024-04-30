# gpu device setup
import os
import jsonlines
import numpy as np
from tqdm import tqdm
import ast
from typing import List

from src.data_creation import utils
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
SEED = 42
torch.manual_seed(SEED)


from src.modelling.self_refine import no_feedback_prompt


# helper function to read results
def read_results(file_path):
    with open(file_path, "r") as f:
        results = f.readlines()
    # results = pd.read_json(file_path, lines=True)
    return results


def postprocess_predictions(prediction):
    prediction = prediction.split("\n\nAnswer:")
    # print(prediction)
    if len(prediction) > 1:
        prediction = prediction[1]
    else:
        prediction = prediction[0]
    # print(prediction)
    prediction = prediction.lstrip()
    return prediction


def score_predictions(args, file_path: str):
    from tigerscore import TIGERScorer
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True)  # on GPU
    results = read_results(file_path)
    # print(results)
    # results = utils.jload(file_path)
    eval_results = []
    for result in tqdm(results):
        result = ast.literal_eval(result)
        prediction = postprocess_predictions(result["prediction"])    # prediction
        # print(prediction)
        # instruction = "Answer the given question."
        instruction = """
You are a helpful, respectful and honest assistant. Please answer the given question to the best of your ability.

If you are unsure about an answer, truthfully say "I don't know"
"""
        input_context = result["prompt"]   # prompt question
        # print(instruction)
        hypo_output = prediction
        results = scorer.score([instruction], [hypo_output], [input_context])
        # print(results)
        eval_results.append(results)
        # break

    # save results to jsonl using jsonlines
    with jsonlines.open(f"tigerscore_{args.output_dir}", "w") as writer:
        writer.write_all(eval_results)


def check_errors(file_path):
    file_name = f"tigerscore_{file_path}"
    with jsonlines.open(file_name) as reader:
        results = list(reader)
    # print(results[0])
    agg_score = 0
    hallucination_score = 0
    total_num_errors = 0
    count = 0
    for result in results:
        result = result[0]
        # print(result)
        if result["score"]:
            score = -result["score"]
            agg_score += score
            num_errors = result["num_errors"]
            hallucination_score += score/num_errors
            total_num_errors += num_errors
            count += 1
    if hallucination_score > 0:
        avg_score = hallucination_score / len(results)
    else:
        avg_score = 0
    print("Total no. of hallucinated samples: ", (count/len(results)) * 100)
    print("Total number of errors: ", total_num_errors)
    print("hallucination score: ", avg_score)


def avg_hallucination_score(scores: List):
    """
    Caclulate the mean and std deviation of hallucination scores
    :param scores:
    :return:
    """
    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std


def calculate_error_correction_metrics(true_labels, predicted_labels):
    corrected_true_positives = sum(
        1 for true_label, pred_label in zip(true_labels, predicted_labels) if true_label == 0 and pred_label == 1)
    uncorrected_true_negatives = sum(
        1 for true_label, pred_label in zip(true_labels, predicted_labels) if true_label == 0 and pred_label == 0)
    uncorrected_false_negatives = sum(
        1 for true_label, pred_label in zip(true_labels, predicted_labels) if true_label == 1 and pred_label == 0)

    # Calculate Precision for error correction
    precision = corrected_true_positives / (corrected_true_positives + uncorrected_false_negatives) \
                if (corrected_true_positives + uncorrected_false_negatives) > 0 else 0

    # Calculate Recall for error correction
    recall = corrected_true_positives / (corrected_true_positives + uncorrected_true_negatives) \
             if (corrected_true_positives + uncorrected_true_negatives) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def calculate_precision_recall(true_labels, predicted_labels):
    # Count the number of true positives (correctly identified errors)
    # model correctly identifies and corrects the errors
    true_positives = sum(
        1 for true_label, pred_label in zip(true_labels, predicted_labels) if true_label == 0 and pred_label == 1)

    # Count the number of false positives (incorrectly identified errors)
    # model identifies errors that are not present in the reference
    false_positives = sum(
        1 for true_label, pred_label in zip(true_labels, predicted_labels) if true_label == 1 and pred_label == 0)

    # Count the number of false negatives (errors that the model failed to correct)
    false_negatives = sum(
        1 for true_label, pred_label in zip(true_labels, predicted_labels) if true_label == 0 and pred_label == 0)

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall


if __name__ == '__main__':
    # add arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, default="",
                        help="path to the output directory")
    parser.add_argument("--score", action="store_true", default=False,
                        help="whether to score the predictions")
    parser.add_argument("--f1_score", action="store_true", default=False,
                        help="whether to score the predictions")
    parser.add_argument("--dataset", type=str, required=True, default="",
                        help="path to the dataset")
    parser.add_argument("--seed", type=int, required=True, default=42,
                        help="random seed for reproducibility")

    args = parser.parse_args()
    file_path = f"experiments/results_{args.output_dir}"
    # file_path = f"results/llama2_13b_no_feedback_responses_{args.dataset}_seed_{args.seed}_all.json"
    # file_path = f"src/data/annotated_data/eli5_errors_complete_1.jsonl"
    # print(results[0])
    if args.score:
        check_errors(args.output_dir)
    elif args.f1_score:
        # calculate f1 score
        # read base file
        base_file = f"tigerscore_llama2_13b_baseline_responses_{args.dataset}_seed_42_all.jsonl"
        with jsonlines.open(base_file) as reader:
            results = list(reader)

        references, predictions = [], []
        references.extend(0 if result[0]["score"] else 1 for result in results)
        # print(references)
        pred_file = f"tigerscore_{args.output_dir}"
        with jsonlines.open(pred_file) as reader:
            preds = list(reader)
        predictions.extend(0 if result[0]["score"] else 1 for result in preds)
        # print(predictions)
        precision, recall, f1 = calculate_error_correction_metrics(references, predictions)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1)
    else:
        score_predictions(args, file_path)


        # predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        # references = [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
        # precision, recall, f1 = calculate_precision_recall_f1(predictions, references)
        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1 score: ", f1)

    # hallucination_scores = [0.0, 0, 1.96]
    # mean, std = avg_hallucination_score(hallucination_scores)
    # print("Mean hallucination score: ", mean)
    # print("Std deviation hallucination score: ", std)
