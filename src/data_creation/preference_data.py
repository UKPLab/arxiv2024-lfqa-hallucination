import pandas as pd
import datasets
from datasets import load_dataset, Dataset, NamedSplit


def create_preference_dataset(file_path: str):
    df = pd.read_csv(file_path, delimiter="\t", index_col=0)
    print(df.head())
    print(df.columns)
    # add a column for preference
    df["preferred_response"] = df.apply(lambda row: row.ans1_text if row.ans_preference == "Answer 1" else row.ans2_text,
                                        axis=1)
    df["rejected_response"] = df.apply(lambda row: row.ans1_text if row.ans_preference == "Answer 2" else row.ans2_text,
                                       axis=1)
    print(df.preferred_response.head())
    print(df.rejected_response.head())
    columns = ["source_file", "question_text", "preferred_response", "rejected_response"]
    df = df[columns]
    df.to_csv("src/data/annotated_data/preference_data_13_03.csv", sep="\t")
    # dataset = Dataset.from_pandas(df, split=NamedSplit("train"))
    # print(dataset)


if __name__ == '__main__':
    file_path = "src/data/annotated_data/complete_data_scores.csv"
    create_preference_dataset(file_path)
