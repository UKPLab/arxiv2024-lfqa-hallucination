from typing import List
import pandas as pd
import ast
import numpy as np

import nltk
nltk.download('punkt')

from src.data_creation import utils


class ResponseScorer:
    def __init__(self, data_path: str, category: str):
        self.data_path = data_path
        self.category = category

    def _load_data(self):
        if self.category is None:
            path = f"{self.data_path}/complete_data_scores.csv"
        else:
            path = f"{self.data_path}/{self.category}/processed_data.csv"
        df = pd.read_csv(path, delimiter="\t", index_col=0)
        return df

    def scorer(self, aspects: List):
        df = self._load_data()

        for row in df.itertuples():
            # print(row)
            # model_score, human_score = 0, 0
            ans1_label, ans2_label = row.ans1_label, row.ans2_label
            # Tokenize the text into sentences
            ans1_len = len(nltk.sent_tokenize(row.ans1_text))
            ans2_len = len(nltk.sent_tokenize(row.ans2_text))

            label_mapping = {
                "Answer 1": ans1_label,
                "Answer 2": ans2_label,
                "answer1": ans1_label,
                "answer2": ans2_label
            }
            len_mapping = {
                "answer1": ans1_len,
                "answer2": ans2_len
            }
            score_mapping = {
                "model_answer": "overall_model_score",
                "human_answer": "overall_human_score",
            }
            scores = {
                "ques_misconception_model_score": 0,
                "ques_misconception_human_score": 0,
                "factuality_model_score": 0,
                "factuality_human_score": 0,
                "relevance_model_score": 0,
                "relevance_human_score": 0,
                "completeness_model_score": 0,
                "completeness_human_score": 0,
                "reference_model_score": 0,
                "reference_human_score": 0,
                "ans_preference_model_score": 0,
                "ans_preference_human_score": 0,
                "overall_model_score": 0,
                "overall_human_score": 0
            }

            for aspect in aspects:
                if aspect != "ans_preference":
                    aspect += "_label"
                value = getattr(row, aspect)

                if aspect == "ans_preference":
                    selected_label = label_mapping.get(value, None)
                    score_key = score_mapping.get(selected_label, None)
                    if score_key is not None:
                        scores[score_key] += 1
                        scores[f"{aspect}_{score_key.split('_')[1]}_score"] = 1
                elif aspect in ["factuality_label", "irrelevance_label"]:
                    if aspect == "factuality_label":
                        scores["factuality_model_score"] = 1
                        scores["factuality_human_score"] = 1
                        name = "factuality"
                    elif aspect == "irrelevance_label":
                        scores["relevance_model_score"] = 1
                        scores["relevance_human_score"] = 1
                        name = "relevance"

                    # scores[f"{aspect.split('_')[0]}_model_score"] = 1
                    # scores[f"{aspect.split('_')[0]}_human_score"] = 1
                    scores["overall_model_score"] += 1
                    scores["overall_human_score"] += 1
                    # handle cases when there is no error
                    if isinstance(value, float):
                        continue

                    selected_spans = ast.literal_eval(getattr(row, f"{aspect.rsplit('_', 1)[0]}_span"))
                    # correct the text
                    selected_spans = [utils.correct_text(span) for span in selected_spans]

                    for i, val in enumerate(ast.literal_eval(value)):
                        selected_label = label_mapping.get(val, None)
                        # select span for the selected label
                        selected_span = selected_spans[i]
                        # get the no. of sentences with nltk of the selected span
                        selected_span_len = len(nltk.sent_tokenize(selected_span))
                        selected_ans_len = len_mapping.get(val, None)
                        score_key = score_mapping.get(selected_label, None)
                        if score_key is not None:
                            scores[score_key] -= selected_span_len/selected_ans_len
                            scores[f"{name}_{score_key.split('_')[1]}_score"] -= selected_span_len/selected_ans_len
                # elif aspect == "incomplete_ans_label":
                #     scores["completeness_model_score"] = 1
                #     scores["completeness_human_score"] = 1
                #     scores["overall_model_score"] += 1
                #     scores["overall_human_score"] += 1
                #
                #     if isinstance(value, float):
                #         continue
                #
                #     answers = []
                #     for val in ast.literal_eval(value):
                #         selected_label = label_mapping.get(val, None)
                #         answers.append(selected_label)
                #     # deduplicate answers
                #     # print(answers)
                #     answers = list(set(answers))
                #     # if answers has key in score_mapping, then subtract 1 to the score
                #     for ans in answers:
                #         score_key = score_mapping.get(ans, None)
                #         print(score_key)
                #         if score_key is not None:
                #             scores[score_key] -= 1
                #             scores[f"completeness_{score_key.split('_')[1]}_score"] -= 1
                elif aspect == "incomplete_ans_label":
                    scores["completeness_model_score"] = 1
                    scores["completeness_human_score"] = 1
                    scores["overall_model_score"] += 1
                    scores["overall_human_score"] += 1
                    name = "completeness"

                    if isinstance(value, float):
                        continue

                    selected_spans = ast.literal_eval(getattr(row, f"{aspect.rsplit('_', 1)[0]}_span"))
                    # correct the text
                    selected_spans = [utils.correct_text(span) for span in selected_spans]

                    for i, val in enumerate(ast.literal_eval(value)):
                        # print(val)
                        selected_label = label_mapping.get(val, None)
                        # select span for the selected label
                        selected_span = selected_spans[i]
                        # print(selected_span)
                        selected_ans_len = len_mapping.get(val, None)
                        if selected_span.replace(":", "").strip() in ["ANSWER1", "ANSWER2"]:
                            selected_span_len = selected_ans_len
                        else:
                            # get the no. of sentences with nltk of the selected span
                            selected_span_len = len(nltk.sent_tokenize(selected_span))
                        # print(selected_span_len)

                        score_key = score_mapping.get(selected_label, None)
                        # print(score_key)
                        if score_key is not None:
                            scores[score_key] -= selected_span_len / selected_ans_len
                            scores[f"{name}_{score_key.split('_')[1]}_score"] -= selected_span_len / selected_ans_len
                elif aspect == "reference_example_label":
                    helpful_value = getattr(row, "reference_example_helpful")
                    # if no reference example is provided, then skip
                    if isinstance(value, float):
                        scores["reference_model_score"] = 1
                        scores["reference_human_score"] = 1
                        scores["overall_model_score"] += 1
                        scores["overall_human_score"] += 1
                        continue

                    answers = []
                    for val, helpful in zip(ast.literal_eval(value), ast.literal_eval(helpful_value)):
                        selected_label = label_mapping.get(val, None)

                        if helpful:
                            answers.append(selected_label)
                    answers = list(set(answers))
                    for ans in answers:
                        score_key = score_mapping.get(ans, None)
                        if score_key is not None:
                            scores[score_key] += 1
                            scores[f"reference_{score_key.split('_')[1]}_score"] += 1

                elif aspect == "ques_misconception_label":
                    if isinstance(value, str):  # misconception span is marked
                        print(value)
                        scores["ques_misconception_model_score"] = 0
                        scores["ques_misconception_human_score"] = 0
                    else:
                        scores["ques_misconception_model_score"] = 1
                        scores["ques_misconception_human_score"] = 1

            scores = {key: round(value, 2) for key, value in scores.items()}
            # append scores to the dataframe
            columns = [
                "ques_misconception_model_score",
                "ques_misconception_human_score",
                "factuality_model_score",
                "factuality_human_score",
                "relevance_model_score",
                "relevance_human_score",
                "completeness_model_score",
                "completeness_human_score",
                "reference_model_score",
                "reference_human_score",
                "ans_preference_model_score",
                "ans_preference_human_score",
                "overall_model_score",
                "overall_human_score"
            ]

            for column in columns:
                df.at[row.Index, column] = scores[column]


        # # save the dataframe
        df.to_csv(f"{self.data_path}/complete_data_scores1.csv", sep="\t")
        # # average the scores for model and human across the dataframe
        model_score = round(df["overall_model_score"].mean(), 2)
        human_score = round(df["overall_human_score"].mean(), 2)

        return model_score, human_score


if __name__ == '__main__':
    aspects = ["ques_misconception", "reference_example", "ans_preference", "factuality", "irrelevance", "incomplete_ans"]
    base_path = "src/data/annotated_data"
    # aspects = ["irrelevance"]
    category = None  # or any particular category
    # categories = ["biology", "chemistry", "economics", "history", "law", "physics", "technology"]
    categories = [None]
    model_scores, human_scores = [], []
    for category in categories:
        print(category)
        scorer = ResponseScorer(
            data_path=base_path,
            category=category,
        )
        model_score, human_score = scorer.scorer(aspects=aspects)
        print(model_score, human_score)
        model_scores.append(model_score)
        human_scores.append(human_score)

    print(model_scores)
    print(human_scores)
    print(np.mean(model_scores))
    print(np.mean(human_scores))
    print(np.std(model_scores))
    print(np.std(human_scores))
