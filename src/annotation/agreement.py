import os
import pandas as pd
import numpy as np
from typing import List

from statsmodels.stats.inter_rater import fleiss_kappa
from krippendorff import alpha


def fleissKappa(responses: List[List]):
    response_array = np.array(responses)
    kappa, _ = fleiss_kappa(response_array)
    return kappa


def annotator_agreement(category=None):
    def convert_answers(response):
        if response == "Answer 1":
            return 1
        else:
            return 2

    dataframes = []
    for annotator in range(1, 4):
        data_path = f"src/data/prolific/results_{category}_tud_{annotator}"
        df = pd.read_csv(
            os.path.join(data_path, "lfqa_pilot_complete.csv"),
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
    # Now you have the merged DataFrame with unique 'id' values

    grouped_df = merged_df.groupby('source_file')[['ans1_label', 'ans2_label', 'ans_preference']].agg(list).reset_index()
    # Count the occurrences of each 'source id'
    source_id_counts = merged_df['source_file'].value_counts()
    # pd.set_option('display.max_rows', None)
    # Filter out 'source id' values that appeared only once
    grouped_df_filtered = grouped_df[grouped_df['source_file'].isin(source_id_counts[source_id_counts > 1].index)]
    responses = grouped_df_filtered["ans_preference"].tolist()
    return responses


if __name__ == '__main__':
    response = annotator_agreement(
        category="economics"
    )
    # print(response)

    response_array = np.array(response)
    alpha_value = alpha(reliability_data=response_array)
    print("Krippendorff's alpha:", alpha_value)
