import json
from typing import Any, Dict, List, Union, Tuple

import pandas as pd
import datasets

from src.data_creation import utils
from src.modelling.chat_templates import llm_prompts


def load_data(data_path: str) -> List[Dict[str, Any]]:
    # convert to hf dataset
    list_data_dict = utils.jload(data_path)
    # convert to pandas dataframe
    df = pd.DataFrame(list_data_dict)

    dataset = datasets.Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)["test"]

    original_columns = dataset.column_names

    def map_dataset(dataset):
        return dataset.map(
            lambda x:
            {
                "prompt": llm_prompts.create_llama_base_prompt(has_input=True).format_map(x),
                "output": x["output"],
            },
            # batched=True,
            remove_columns=original_columns,
        )

    test_data = map_dataset(dataset)
    test_data = test_data.shuffle(seed=42)

    return test_data


def load_predictions(predictions_path: str) -> Tuple[List, List]:
    list_data_dict = utils.jload(predictions_path)
    # convert to pandas dataframe
    df = pd.DataFrame(list_data_dict)
    predictions = df["prediction"].tolist()
    consistency = df["consistency"].tolist()
    return predictions, consistency


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, default="src/data/annotated_data/incomplete_ans_detection_data_2.jsonl"
    )
    parser.add_argument(
        "--pred_file_path", type=str,
        default="results/llama2_13b_completeness_feedback_responses_held_out_seed_42.jsonl"
    )
    args = parser.parse_args()

    results = load_data(args.file_path)
    gold_answers = []
    for result in results:
        output = result["output"]
        sentences = output.split("\n")

        incomplete_sent_ids = []
        for i, sent in enumerate(sentences):
            if sent.__contains__("[Incomplete]"):
                incomplete_sent_ids.append(i+1)
        gold_answers.append(incomplete_sent_ids)
    print(gold_answers)

    predictions, consistency = load_predictions(args.pred_file_path)
    # print(predictions)
    pred_answers = []
    for i, pred in enumerate(predictions):
        sentences = pred.split("\n")
        incomplete_sent_ids = []
        for j, sent in enumerate(sentences):
            if sent.__contains__("[Incomplete]"):
                incomplete_sent_ids.append(j+1)
        pred_answers.append(incomplete_sent_ids)
    print(pred_answers)

    correct, incorrect = 0, 0
    c_count, i_count = 0, 0
    num_samples = len(gold_answers)
    correct_consistency_score, incorrect_consistency_score = 0, 0
    for gold, pred, score in zip(gold_answers, pred_answers, consistency):
        for p in pred:
            if p in gold or p+1 in gold or p-1 in gold:
                correct += 1
                correct_consistency_score += score
                c_count += 1
                break
            else:
                incorrect += 1
                incorrect_consistency_score += score
                i_count += 1
                break

    print(correct, incorrect, num_samples)
    accuracy = (correct / num_samples) * 100
    print(accuracy)
    consistency_score = correct_consistency_score / c_count
    print(consistency_score)
    i_consistency_score = incorrect_consistency_score / i_count
    print(i_consistency_score)
