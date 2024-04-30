import pandas as pd
import ast
from utils import correct_text
import re
import json
from typing import List

from src.data_creation import utils
# avoid future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

TASK_INSTRUCTION = {
    "incomplete_ans": "Given a question-answer pair, evaluate whether the answer sufficiently "
                      "addresses the question. If the answer provides all the necessary "
                      "information to answer the question, return a score of 1.0; otherwise, return 0.0.",
    # "incomplete_ans": "Given a question-answer pair, evaluate whether the answer sufficiently "
    #                   "addresses the question. If the answer provides all the necessary "
    #                   "information to answer the question, return the label [Complete]; otherwise, return "
    #                   "the label [Incomplete].",
}


class ErrorDataset:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        """
        Load annotated data
        :return:
        """
        df = pd.read_csv(
            self.data_path,
            na_values=['NA', 'NaN', '', 'NULL', 'missing', "[]"],
            delimiter="\t",
        )
        df.reset_index(drop=True, inplace=True)
        return df

    def _get_correct_answers(self, example, answer_choice):
        """
        Get the error-free answers for the given aspect, answer choice (human or model)
        :param example: example row
        :param answer_choice: randomized answer choice
        :return:
        """

        if example["ans1_label"].__contains__(answer_choice):
            answer = example["ans1_text"]
            label = "answer1"  # example["ans1_label"]

        elif example["ans2_label"].__contains__(answer_choice):
            answer = example["ans2_text"]
            label = "answer2"  # example["ans2_label"]
        else:
            return None, None
        return answer, label

    def create_tags(
            self,
            aspect: str,
            answer_choice: str,
            use_all_data: bool = False,
            max_correct_answers: int = None
    ):
        """
        Create the highlighted identifier tags for the given aspects
        :param aspect:
        :param answer_choice:
        :param add_score:
        :return:
        """
        df = self.load_data()
        complete_answers_count = 0
        incomplete_answers_count = 0
        unknown_answers_count = 0

        if max_correct_answers is None:
            max_correct_answers = df.shape[0]

        for i, ex in df.iterrows():
            gold_answer, label = self._get_correct_answers(ex, answer_choice)
            # print(ex[f"{aspect}_label"])
            # print(label)
            if pd.isna(ex[f"{aspect}_label"]) or label not in ex[f"{aspect}_label"]:
                response = 1
                if not use_all_data:
                    continue
                if complete_answers_count >= max_correct_answers:
                    continue
                if gold_answer is None:
                    continue
                complete_answers_count += 1
            elif label in ex[f"{aspect}_label"]:
                incomplete_answers_count += 1
                response = 0
            else:
                continue

            df.loc[i, f"{aspect}_label_binary"] = response
            df.loc[i, "identified_answer"] = gold_answer
            # print("*" * 50)
            # print(gold_answer)

        cols = [f"{aspect}_label_binary"]
        df.dropna(subset=cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # filter columns
        final_cols = ["question_text", "identified_answer"] + cols
        df = df[final_cols]
        # shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # print(df.head())
        print(complete_answers_count)
        print(incomplete_answers_count)
        print(unknown_answers_count)
        return df


if __name__ == '__main__':
    data_path = "src/data/annotated_data/complete_data_scores.csv"
    dataset = ErrorDataset(data_path)
    aspect = "incomplete_ans"  # "irrelevance", "incomplete_ans", "reference_example"
    use_all_data = True
    use_reason = False
    add_score = False
    max_correct_answers = None

    # write the question answers to a json file
    examples = []

    complete_df = pd.DataFrame()
    for ans_choice in ["model", "human"]:
        df = dataset.create_tags(
            aspect=aspect,
            answer_choice=ans_choice,
            use_all_data=use_all_data,
            max_correct_answers=max_correct_answers
        )

        # concatenate the dfs for different answer choices
        if complete_df.empty:
            complete_df = df
        else:
            complete_df = pd.concat([complete_df, df], axis=0, ignore_index=True)
    complete_df = complete_df.loc[:, ~complete_df.columns.str.contains('^Unnamed')]
    print(complete_df.shape)
    for i, row in complete_df.iterrows():
        examples.append({
            "instruction": f"{TASK_INSTRUCTION[aspect]}",
            "input": f"Question: {row['question_text']}\nAnswer: {row['identified_answer']}",
            "output": row[f"{aspect}_label_binary"],
        })

    utils.jdump(examples, f"src/data/annotated_data/{aspect}_detection_binary.jsonl")
