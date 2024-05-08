import json
from typing import Any, Dict, List, Union, Tuple
import torch
import pandas as pd
import datasets
import statistics
from sentence_transformers import SentenceTransformer, util

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


def calculate_bertscore(predictions: List[str], references: List[str]) -> Dict[str, Union[float, List[float]]]:
    from evaluate import load
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli"
    )
    return results


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
    parser.add_argument(
        "--mode", type=str,
        default="different",
        help="exact, adjacent, different"
    )

    args = parser.parse_args()

    results = load_data(args.file_path)
    predictions, consistency = load_predictions(args.pred_file_path)

    gold_spans, gold_answers = [], []
    for result in results:
        output = result["output"]
        sentences = output.split("\n")

        incomplete_sent_ids, incomplete_reasons = [], []
        for i, sent in enumerate(sentences):
            if sent.__contains__("[Incomplete]"):
                incomplete_sent_ids.append(i+1)
                incomplete_reasons.append(sent.split("Reasons: ")[1])
        gold_spans.append(incomplete_sent_ids)
        gold_answers.append(" ".join(incomplete_reasons))
    print(gold_spans)

    # print(predictions)
    pred_spans, pred_answers = [], []
    for i, pred in enumerate(predictions):
        sentences = pred.split("\n")
        incomplete_sent_ids, incomplete_reasons = [], []
        for j, sent in enumerate(sentences):
            if sent.__contains__("[Incomplete]"):
                incomplete_sent_ids.append(j+1)
                incomplete_reasons.append(sent.split("Reasons: ")[1])
        pred_spans.append(incomplete_sent_ids)
        pred_answers.append(" ".join(incomplete_reasons))
    print(pred_spans)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    exact, adjacent, different = 0, 0, 0
    e_count, a_count, d_count = 0, 0, 0
    num_samples = len(gold_answers)
    e_consistency_score, a_consistency_score, d_consistency_score = 0, 0, 0
    e_bertscore, a_bertscore, d_bertscore = 0, 0, 0
    e_cos_sim, a_cos_sim, d_cos_sim = 0, 0, 0
    for gold, pred, score, gold_ans, pred_ans in zip(gold_spans, pred_spans, consistency, gold_answers, pred_answers):
        for p in pred:
            if p in gold:
                exact += 1
                e_consistency_score += score
                # Compute embedding for both lists
                embedding_1 = model.encode(pred_ans, convert_to_tensor=True)
                embedding_2 = model.encode(gold_ans, convert_to_tensor=True)
                e_cos_sim += util.pytorch_cos_sim(embedding_1, embedding_2).item()
                # e_bertscore += calculate_bertscore([pred_ans], [gold_ans])["f1"][0]

                e_count += 1
                break
            elif p+1 in gold or p-1 in gold:
                adjacent += 1
                a_consistency_score += score
                embedding_1 = model.encode(pred_ans, convert_to_tensor=True)
                embedding_2 = model.encode(gold_ans, convert_to_tensor=True)
                a_cos_sim += util.pytorch_cos_sim(embedding_1, embedding_2).item()
                # a_bertscore += calculate_bertscore([pred_ans], [gold_ans])["f1"][0]
                a_count += 1
                break
            else:
                different += 1
                d_consistency_score += score
                embedding_1 = model.encode(pred_ans, convert_to_tensor=True)
                embedding_2 = model.encode(gold_ans, convert_to_tensor=True)
                d_cos_sim += util.pytorch_cos_sim(embedding_1, embedding_2).item()
                # d_bertscore += calculate_bertscore([pred_ans], [gold_ans])["f1"][0]
                d_count += 1
                break

    # print(correct, incorrect, num_samples)
    if args.mode == "exact":
        print("Exact Match")
        accuracy = (exact / num_samples) * 100
        consistency_score = e_consistency_score / e_count
        sbert_score = e_cos_sim / e_count
        # bertscore = e_bertscore / e_count
    elif args.mode == "adjacent":
        print("Adjacent Match")
        accuracy = (adjacent / num_samples) * 100
        consistency_score = a_consistency_score / a_count
        sbert_score = a_cos_sim / a_count
        # bertscore = a_bertscore / a_count
    else:
        print("Different Match")
        accuracy = (different / num_samples) * 100
        consistency_score = d_consistency_score / d_count
        sbert_score = d_cos_sim / d_count
        # bertscore = d_bertscore / d_count

    print(f"Accuracy: {accuracy}")
    print(f"Consistency Score: {consistency_score}")
    # print(f"BertScore: {bertscore}")
    print(f"Sentence Bert Score: {sbert_score}")
