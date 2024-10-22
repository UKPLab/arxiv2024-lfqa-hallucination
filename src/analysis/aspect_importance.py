"""
determine the aspect importance as judged by human annotators
"""

import pandas as pd
from typing import List
import ast
import os


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
        if aspect == "ques_misconception":
            # count entries where the aspect label is not None
            # here we do not care about the answer label
            ans1_count = df[f"{aspect}_label"].notna().sum()
            # print(ans1_count)
            ans2_count = 0
        elif aspect == "reference_example":
            ans1_count = ((df["ans1_label"] == answer_label) & (
                df.apply(lambda x: "answer1" in x[f"{aspect}_label"] and
                                   ast.literal_eval(x[f"{aspect}_helpful"])[ast.literal_eval(x[f"{aspect}_label"]).index("answer1")] is False, axis=1))).sum()
            ans2_count = ((df["ans2_label"] == answer_label) & (
                df.apply(lambda x: "answer2" in x[f"{aspect}_label"] and
                                   ast.literal_eval(x[f"{aspect}_helpful"])[ast.literal_eval(x[f"{aspect}_label"]).index("answer2")] is False, axis=1))).sum()
        else:
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
        for aspect in self.aspects:
            # print(aspect)
            # if aspect == "ques_misconception":
            #     columns = [f"{aspect}_label", "ans1_label", "ans2_label"]
            #     continue
            if aspect == "reference_example":
                columns = [f"{aspect}_label", "ans1_label", "ans2_label", "reference_example_helpful"]
            else:
                columns = [f"{aspect}_label", "ans1_label", "ans2_label"]
            filtered_df = df[columns]
            # print(filtered_df.head())
            filtered_df = filtered_df[filtered_df[f"{aspect}_label"].notna()]
            # print(filtered_df.shape)
            filtered_df = filtered_df[
                filtered_df[f"{aspect}_label"].apply(lambda x: len(ast.literal_eval(x)) > 0)]
            # print(filtered_df)
            human_counts = self.count_answers(filtered_df, aspect, "human_answer")
            model_counts = self.count_answers(filtered_df, aspect, "model_answer")
            # print("Aspect: ", aspect)
            # print("Human counts: ", human_counts)
            # print("Model counts: ", model_counts)
            # print("--"*8)
            # break
            aspect_wise_annotation[aspect] = {"human": human_counts, "model": model_counts}
        # print(aspect_wise_annotation)

        # Calculate the total human and model counts for all aspects
        total_human_counts = sum(aspect_counts['human'] for aspect_counts in aspect_wise_annotation.values())
        total_model_counts = sum(aspect_counts['model'] for aspect_counts in aspect_wise_annotation.values())
        human_importance, model_importance = {}, {}

        # Calculate and print the human and model agreement percentages for each aspect
        for aspect, aspect_counts in aspect_wise_annotation.items():
            human_percentage = (aspect_counts["human"] / total_human_counts) * 100
            if total_model_counts == 0:
                model_percentage = 0
            else:
                model_percentage = (aspect_counts["model"] / total_model_counts) * 100

            # print("Aspect: {}".format(aspect))
            # print("Human Percentage: {:.2f}%".format(human_percentage))
            # print("Model Percentage: {:.2f}%".format(model_percentage))
            # print()
            human_importance[aspect] = human_percentage
            model_importance[aspect] = model_percentage

        return human_importance, model_importance


if __name__ == '__main__':
    aspects = ["ques_misconception", "factuality", "irrelevance", "incomplete_ans", "reference_example"]
    categories = ["biology", "chemistry", "economics", "history", "law", "physics", "technology"]
    base_path = "src/data/annotated_data"
    files = os.listdir(base_path)

    for category in categories:
        result = {"human": {}, "model": {}}  # store the results for each aspect
        data_path = f"{base_path}/{category}"
        imp = AspectImportance(
            data_path=os.path.join(data_path, "processed_data.csv"),
            aspects=aspects,
        )
        human_imp, model_imp = imp.analyze_fine_grained_importance()
        # print(human)

        for aspect, imp in human_imp.items():
            if aspect in result:
                result["human"][aspect] += imp
            else:
                result["human"][aspect] = imp
        for aspect, imp in model_imp.items():
            if aspect in result:
                result["model"][aspect] += imp
            else:
                result["model"][aspect] = imp
        # make all result values to 2 decimal places
        for key, value in result.items():
            for aspect, imp in value.items():
                result[key][aspect] = round(imp, 2)
        print(category)
        print(result)
        print("--"*8)
