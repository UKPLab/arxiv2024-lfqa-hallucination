"""
determine the aspect importance as judged by human annotators
"""

import pandas as pd
from typing import List
import ast


class AspectImportance:
    def __init__(self, data_path: str, aspects: List):
        self.data_path = data_path
        self.aspects = aspects

    def _load_data(self):
        df = pd.read_csv(self.data_path, sep="\t", index_col=0)
        return df

    def analyze_overall_importance(self):
        """
        document level analysis of marked aspects
        :return:
        """
        df = self._load_data()
        column_wise_annotation = {}
        for aspect in aspects:
            column = f"{aspect}_span"
            filtered_df = df[df[column].notna() &
                             df[column].apply(lambda x: len(x) > 0 if isinstance(x, list) else True)]
            filtered_df = filtered_df[[column]]
            column_wise_annotation[column] = int(filtered_df.count())

        total_span = sum(column_wise_annotation.values())
        percentage_data = {key: (value / total_span) * 100 for key, value in column_wise_annotation.items()}

        for key, value in percentage_data.items():
            print(f'{key}: {value:.2f}%')

    def count_answers(self, df, aspect, answer_label):
        ans1_count = ((df["ans1_label"] == answer_label) & (
            df[f"{aspect}_label"].apply(lambda x: "answer1" in x))).sum()
        ans2_count = ((df["ans2_label"] == answer_label) & (
            df[f"{aspect}_label"].apply(lambda x: "answer2" in x))).sum()
        return ans1_count + ans2_count

    def analyze_fine_grained_importance(self):
        """
        human/model level analysis of marked aspects
        :return:
        """
        df = self._load_data()
        aspect_wise_annotation = {}
        for aspect in aspects:
            if aspect == "ques_misconception":
                continue
            columns = [f"{aspect}_label", "ans1_label", "ans2_label"]
            filtered_df = df[columns]
            filtered_df = filtered_df[
                filtered_df[f"{aspect}_label"].apply(lambda x: len(ast.literal_eval(x)) > 0)]
            # print(filtered_df)
            human_counts = self.count_answers(filtered_df, aspect, "human_answer")
            model_counts = self.count_answers(filtered_df, aspect, "model_answer")
            print("Aspect: ", aspect)
            print("Human counts: ", human_counts)
            print("Model counts: ", model_counts)
            print("--"*8)
            # break
            aspect_wise_annotation[aspect] = {"human": human_counts, "model": model_counts}
        print(aspect_wise_annotation)

        # Calculate the total human and model counts for all aspects
        total_human_counts = sum(aspect_counts['human'] for aspect_counts in aspect_wise_annotation.values())
        total_model_counts = sum(aspect_counts['model'] for aspect_counts in aspect_wise_annotation.values())

        # Calculate and print the human and model agreement percentages for each aspect
        for aspect, aspect_counts in aspect_wise_annotation.items():
            human_percentage = (aspect_counts["human"] / total_human_counts) * 100
            model_percentage = (aspect_counts["model"] / total_model_counts) * 100

            print("Aspect: {}".format(aspect))
            print("Human Percentage: {:.2f}%".format(human_percentage))
            print("Model Percentage: {:.2f}%".format(model_percentage))
            print()


if __name__ == '__main__':
    aspects = ["ques_misconception", "factuality", "irrelevance", "incomplete_ans", "reference_example"]
    category = "history"
    num_annotator = 3
    prolific_id = "613637a4f7a0e5359082010b"
    imp = AspectImportance(
        data_path=f"src/data/prolific/results_{category}_tud_{num_annotator}_{prolific_id}/lfqa_pilot_complete.csv",
        aspects=aspects,
    )
    imp.analyze_fine_grained_importance()
