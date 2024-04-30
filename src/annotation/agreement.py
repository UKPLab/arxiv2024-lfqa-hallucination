import os
import pandas as pd
import numpy as np
from typing import List

import statistics

from statsmodels.stats.inter_rater import fleiss_kappa
from krippendorff import alpha


def fleissKappa(responses: List[List]):
    response_array = np.array(responses)
    kappa, _ = fleiss_kappa(response_array)
    return kappa


def calculate_percentage(row):
    preference = int(row['majority_preference'])  # Convert preference to an integer
    if preference == 1:
        answer = row['ans1_label'][0]
    else:
        answer = row['ans2_label'][0]
    return answer


def process_annotator_data(category=None, num_annotators=3):
    def convert_answers(response):
        if response == "Answer 1":
            return 1
        else:
            return 2

    dataframes = []
    for annotator in range(1, num_annotators+1):
        base_path = f"src/data/annotated_data/{category}"

        df = pd.read_csv(
            os.path.join(base_path, "processed_data.csv"),
            na_values=['NA', 'NaN', '', 'NULL', 'missing', "[]"],
            delimiter="\t",
        )
        df.reset_index(drop=True, inplace=True)
        # df['ans_preference'] = df['ans_preference'].apply(eval)  # Convert string lists to actual lists
        df['ans_preference'] = df['ans_preference'].apply(convert_answers)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    # merged_df = merged_df.drop_duplicates(subset=["source_file"])
    merged_df.drop(columns=['Unnamed: 0'], inplace=True)
    return merged_df


def process_data_for_agreement(category=None, num_annotators=3):
    def convert_answers(response):
        if response == "Answer 1":
            return 1
        else:
            return 2

    dataframes = []
    for annotator in range(1, num_annotators+1):
        base_path = f"src/data/annotated_data/{category}"
        files = os.listdir(base_path)
        data_path = f"results_{category}_tud_{annotator}"
        for file in files:
            if file.__contains__(data_path):
                data = f"{base_path}/{file}"
                break

        df = pd.read_csv(
            os.path.join(data, "lfqa_pilot_complete.csv"),
            na_values=['NA', 'NaN', '', 'NULL', 'missing', "[]"],
            delimiter="\t",
        )
        df.reset_index(drop=True, inplace=True)
        # df['ans_preference'] = df['ans_preference'].apply(eval)  # Convert string lists to actual lists
        df['ans_preference'] = df['ans_preference'].apply(convert_answers)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    # merged_df = merged_df.drop_duplicates(subset=["source_file"])
    merged_df.drop(columns=['Unnamed: 0'], inplace=True)
    return merged_df


def calculate_preference(category=None, num_annotators=3):
    data = process_annotator_data(category, num_annotators)
    grouped_df = data.groupby('source_file')[['ans1_label', 'ans2_label', 'ans_preference']].agg(list).reset_index()
    print(grouped_df.head())
    print(grouped_df.shape)
    # df[['human_percentage', 'model_percentage']] = df.apply(calculate_percentage, axis=1)
    grouped_df['majority_preference'] = grouped_df['ans_preference'].apply(lambda x: statistics.mode(x))
    grouped_df['final_preference'] = grouped_df.apply(calculate_percentage, axis=1)
    print(grouped_df.shape)
    value_counts = dict(grouped_df['final_preference'].value_counts())
    model_percentage = (value_counts['model_answer'] /
                        (value_counts['model_answer'] + value_counts['human_answer'])) * 100
    return model_percentage, 100-model_percentage


def calculate_len(row, check):
    label1 = row['ans1_label'][0]
    label2 = row['ans2_label'][0]

    if label1 == check:
        length = len(row['ans1_text'][0].split())
    elif label2 == check:
        length = len(row['ans2_text'][0].split())
        # print(row['ans2_text'])
        # print(length)
    return length


def calculate_answer_stats(category=None, num_annotators=3):
    data = process_annotator_data(category, num_annotators)
    grouped_df = data.groupby('source_file')[['ans1_text', 'ans2_text', 'ans1_label', 'ans2_label']].agg(list).reset_index()
    grouped_df["human_ans_len"] = grouped_df.apply(calculate_len, args=("human_answer", ), axis=1)
    grouped_df["model_ans_len"] = grouped_df.apply(calculate_len, args=("model_answer", ), axis=1)
    # print(grouped_df.head())
    # human_lens = grouped_df["human_ans_len"].tolist()
    # human_min, human_mean, human_max = min(human_lens), statistics.mean(human_lens), max(human_lens)
    # model_lens = grouped_df["model_ans_len"].tolist()
    # model_min, model_mean, model_max = min(model_lens), statistics.mean(model_lens), max(model_lens)
    # print("human: ", human_min, human_mean, human_max)
    # print("model: ", model_min, model_mean, model_max)
    # return (human_min, human_mean, human_max), (model_min, model_mean, model_max)
    return grouped_df


def calculate_agreement(category=None, num_annotators=3):

    data = process_data_for_agreement(category, num_annotators)
    print(data.head())
    grouped_df = data.groupby('source_file')[['ans1_label', 'ans2_label', 'ans_preference']].agg(list).reset_index()
    print(grouped_df.head())
    # Count the occurrences of each 'source id'
    source_id_counts = data['source_file'].value_counts()
    grouped_df_filtered = grouped_df[grouped_df['source_file'].isin(source_id_counts[source_id_counts > 1].index)]
    # print(grouped_df_filtered.head(n=30))
    responses = grouped_df_filtered["ans_preference"].tolist()
    return responses


if __name__ == '__main__':

    ###########################################################
    # AGREEMENT
    ###########################################################
    response = calculate_agreement(
        category="physics",
        num_annotators=3
    )
    print(response)
    response_array = np.array(response)
    transposed_array = list(map(list, zip(*response_array)))
    print(transposed_array)

    alpha_value = alpha(reliability_data=transposed_array)
    print("Krippendorff's alpha:", alpha_value)

    ############################################################

    ###########################################################
    # PREFERENCE
    ###########################################################
    # categories = ["biology", "chemistry", "economics", "history", "law", "physics", "technology"]
    # for category in categories:
    #     response = calculate_preference(
    #         category=category,
    #         num_annotators=3
    #     )
    #     print(category)
    #     print(response)
    #     print("--"*8)
    ###########################################################
    # STATS
    ###########################################################
    # response = calculate_answer_stats(
    #     category="biology",
    #     num_annotators=3
    # )
