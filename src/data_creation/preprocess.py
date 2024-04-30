import os
import re
import pycountry

import pandas as pd
# pd.set_option('display.max_columns', None)
from src.data_creation import utils


class PreprocessAnnotations:
    def __init__(self, data_path, category):
        self.data_path = data_path
        self.category = category

    def _load_data(self):
        path = f"{self.data_path}/{self.category}"

        # List all files in the directory
        dir_paths = [os.path.join(path, filename) for filename in os.listdir(path) if not filename.endswith(".csv") and
                     not filename.endswith(".jsonl")]

        complete_df = pd.DataFrame()
        for dir in dir_paths:
            df = pd.read_csv(
                os.path.join(dir, "lfqa_pilot_complete.csv"),
                na_values=['NA', 'NaN', '', 'NULL', 'missing', "[]"],
                delimiter="\t",
                index_col=0
            )
            df.reset_index(drop=True, inplace=True)
            # df["annotator_id"] = dir.split("_")[-1]

            complete_df = pd.concat([df, complete_df], axis=0, ignore_index=True)

        return complete_df

    def preprocess(self, selected_annotator, save_path):
        data = self._load_data()
        # identify duplicate docs
        duplicate_docs = data[data["source_file"].duplicated()]["source_file"].unique()

        rows_to_remove = data[data["source_file"].isin(duplicate_docs) &
                              (data["annotator"] != selected_annotator)].index
        filtered_df = data.drop(rows_to_remove).reset_index(drop=True)
        aspects = ["factuality", "irrelevance", "incomplete_ans", "ques_misconception", "reference_example"]
        aspect_cols = [f"{aspect}_span" for aspect in aspects]
        qa_cols = ["question_text", "ans1_text", "ans2_text"]
        columns_to_correct = qa_cols + aspect_cols
        filtered_df = utils.text_correction(filtered_df, columns_to_correct)

        # save data
        filtered_df.to_csv(save_path, sep="\t")


if __name__ == '__main__':
    base_path = "src/data/annotated_data"
    # handpicked annotators to select the best annotator for each category
    selected_annotators = ["4E4KDglOHRxflovIhgyahA", "cq47p0--97gGaUn_1LZktA", "n3-ljRxwKXnMTea1LVfhyQ",
                           "l9ZE9PGM5FSTAxqwc3ad1w", "Desc6S0nEIJt9E3gMwrXEA", "M3fXDYhva6bay7JfLfN3vQ",
                           "kwrgq4M0bY2l3J8r4P7brg"]
    # read annotators from the json file
    annotators = utils.read_json("src/data/annotators.json")
    # read from annotators.json
    processed_file_name = f"processed_data"
    for info in annotators:
        category = info["subject"]
        annotators = list(info["data"].values())
        processor = PreprocessAnnotations(
                data_path=base_path,
                category=category,
            )

        save_path = f"{base_path}/{category}/{processed_file_name}.csv"
        selected_annotator = [annotator for annotator in annotators if annotator in selected_annotators][0]
        processor.preprocess(selected_annotator=selected_annotator, save_path=save_path)
    # merge datasets and save
    complete_dataset = utils.merge_datasets(base_path, file_name=processed_file_name)
    complete_dataset.to_csv(os.path.join(base_path, "complete_data.csv"), sep="\t")
