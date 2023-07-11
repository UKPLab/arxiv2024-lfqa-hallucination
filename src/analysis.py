from inceptalytics import Project
from inceptalytics.utils import annotation_info_from_xmi_zip, SENTENCE_TYPE_NAME
import pandas as pd
from collections import Counter, OrderedDict
import ast
import numpy as np

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def get_annotations(layer=None, feature=None, type_name=None):
    if feature:
        feature_path = f'{layer}>{feature}'
    else:
        feature_path = f'{layer}'

    # source_files = ["00f64ae1-c96a-4b2e-9912-c40e295e2565.txt", "02062ffd-f0bb-4566-847c-fc7057d4ce0b.txt",
    #                 "02408398-02c5-4afc-866d-4225e869d49a.txt", "0242d81a-2c1b-4021-9aa1-c5cfcf38a784.txt",
    #                 "031d790a-7d9d-498e-a36d-a28eaca304ac.txt", "03235d4c-dcb0-4cac-8dc2-5d17b072f5ac.txt",
    #                 "037f220f-7797-4b35-bcf0-201336c4127b.txt", "03804547-6f04-472b-82fc-bf073985dfc2.txt",
    #                 "03fd2c88-e3e9-4ba5-8641-fa7814df6b08.txt", "04ac3833-c4d7-4bcb-a3f4-f133819ef4df.txt",
    #                 "059fa085-f357-478a-83c1-fc5933bbad6a.txt", "05b4333c-1c42-4ede-b62c-830c38cbfea2.txt",
    #                 "06210903-a207-4171-901b-45d89165bec3.txt", "0725d873-e32c-4c8e-bbcf-8e3eb6a8a0cd.txt",
    #                 "075398ee-9f9f-4ff7-97aa-7d0315844130.txt"]
    source_files = [file for file in project.source_file_names if not file.__contains__("completion.txt")]
    print(f"Source files: {source_files}")
    # biology
    # annotators = ['ALDTHTi6bdUV_PxRqESvew', 'ElEWCTeqc0gwYO7Ze9yW0Q', 'frFLqX9g5biFUM6tBMAoFA']
    annotators = ['U3I1rvU52xrOWLEnDh994Q', 'IYb9xb6gp9-A7SPZuG71pA']
    # economics
    # annotators = ['U4U-HsGbDSjUjGsQYqwzCA', 'uCKTljvJIvpxenurEySxgA', 'Qp4YXOnvjo3VVadyDJRMAA']

    # select reduced view
    reduced_annos = project.select(
        annotation=feature_path,
        annotators=annotators,
        source_files=source_files
    )

    # print('# pref. annotations in view:', reduced_annos.count())
    # print('# annotations per file per annotator', reduced_annos.count(grouped_by=['source_file', 'annotator']))

    df = reduced_annos.data_frame
    # print(df.columns)

    if "reference" in type_name:
        # get the reference helpfulness
        df['helpful'] = df['_annotation'].apply(lambda x: x['Isithelpful'])

    # remove the annotation column and sentence column
    df.drop(columns=['sentence', '_annotation'], inplace=True)
    # change column names for readability
    for col in df.columns:
        if col.startswith('source_file') or col.startswith('annotator'):  # keep the original column name for later merge
            continue
        elif col.startswith('annotation'):
            df = df.rename(columns={col: col.replace('annotation', f'{type_name}_reason')})
        elif col.startswith('text'):
            df = df.rename(columns={col: col.replace('text', f'{type_name}_span')})
        else:
            df = df.rename(columns={col: col.replace(col, f'{type_name}_{col}')})

    # convert span, reason, begin, end, sentence text to list of strings
    df[f'{type_name}_span'] = df[f'{type_name}_span'].apply(lambda x: [x])
    df[f'{type_name}_reason'] = df[f'{type_name}_reason'].apply(lambda x: [x])
    df[f'{type_name}_begin'] = df[f'{type_name}_begin'].apply(lambda x: [x])
    df[f'{type_name}_end'] = df[f'{type_name}_end'].apply(lambda x: [x])
    df[f'{type_name}__sentence_text'] = df[f'{type_name}__sentence_text'].apply(lambda x: [x])

    # if source file and annotator are the same, then append the span, reason, begin, end, sentence text
    if "reference" in type_name:
        df[f'{type_name}_helpful'] = df[f'{type_name}_helpful'].apply(lambda x: [x])
        df = df.groupby(['source_file', 'annotator']).agg(
            {f'{type_name}_span': 'sum', f'{type_name}_reason': 'sum', f'{type_name}_begin': 'sum',
             f'{type_name}_end': 'sum', f'{type_name}__sentence_text': 'sum',
             f'{type_name}_helpful': 'sum'}).reset_index()
    else:
        df = df.groupby(['source_file', 'annotator']).agg(
            {f'{type_name}_span': 'sum', f'{type_name}_reason': 'sum', f'{type_name}_begin': 'sum',
             f'{type_name}_end': 'sum', f'{type_name}__sentence_text': 'sum'}).reset_index()

    return df


def layer_wise_analysis(path, layer):
    df = pd.read_csv(path, sep='\t')
    print(df.head())
    choice = []
    ref_helpful = []
    # iterate over the rows
    for index, row in df.iterrows():
        if df[f"{layer}_label"][index].__contains__("1"):
            choice.append(df[f"ans1_label"][index])
            ref_helpful.append(df[f"reference_example_helpful"][index])  # get the reference helpfulness
        elif df[f"{layer}_label"][index].__contains__("2"):
            choice.append(df[f"ans2_label"][index])
            ref_helpful.append(df[f"reference_example_helpful"][index])  # get the reference helpfulness

    # list of lists to list for ref_helpful
    # ref_helpful = [item for sublist in ref_helpful for item in ast.literal_eval(sublist)]
    # # change true to 1 and false to 0
    # ref_helpful = [1 if x else 0 for x in ref_helpful]
    # print(f"Choice: {choice}")
    # print(f"Reference helpful: {ref_helpful}")
    # ref_human = [ref_helpful[idx] for idx, x in enumerate(choice) if "human" in x]
    # ref_human_helpful = [sum(ref_human), len(ref_human) - sum(ref_human)]
    # ref_model = [ref_helpful[idx] for idx, x in enumerate(choice) if "model" in x]
    # ref_model_helpful = [sum(ref_model), len(ref_model) - sum(ref_model)]

    ref_human_helpful = []
    ref_model_helpful = []

    # calculate number of times each choice is selected
    choice_count = Counter(choice)
    print(f"Choice count: {choice_count}")
    # move human choice to start and model choice to end
    human_choice = choice_count.pop("human_answer")
    model_choice = choice_count.pop("model_answer")

    choice_count = OrderedDict([('human_answer', human_choice), ('model_answer', model_choice)])
    print(f"Choice count: {choice_count}")
    # form 2 lists for plotting labels and counts
    labels = []
    counts = []
    for key, value in choice_count.items():
        labels.append(key)
        counts.append(value)
    print(f"Labels: {labels}")
    print(f"Counts: {counts}")

    return labels, counts, ref_human_helpful, ref_model_helpful


# function to get the top k words in the answer_preference_reason in df
def get_top_k_words(path, layer, k):

    df = pd.read_csv(path, sep='\t')
    choice = []
    ans_reason = []

    for index, row in df.iterrows():
        if df[f"{layer}"][index] is not np.nan:
            if df[f"{layer}"][index].__contains__("1"):
                choice.append(df[f"ans1_label"][index])
                ans_reason.append(df[f"ans_preference_reason"][index])  # get the reference helpfulness
            elif df[f"{layer}"][index].__contains__("2"):
                choice.append(df[f"ans2_label"][index])
                ans_reason.append(df[f"ans_preference_reason"][index])  # get the reference helpfulness

    ans_human = [ans_reason[idx] for idx, x in enumerate(choice) if "human" in x]
    ans_model = [ans_reason[idx] for idx, x in enumerate(choice) if "model" in x]


    # # get the answer_preference_reason column
    # ans_pref_reason = df['ans_preference_reason'].tolist()
    # print(f"Answer preference reason: {ans_pref_reason}")
    # get avg length of answer preference reason in words
    avg_len = sum(len(x.split()) for x in ans_human) / len(ans_human)
    # avg_len = sum(len(ast.literal_eval(x)) for x in ans_pref_reason) / len(ans_pref_reason)
    print(f"Average length of answer preference reason: {avg_len}")

    # add to nltk stopwords
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords.extend(['1', '2', 'answer', 'Answer', 'question', 'one', ',', 'I', 'It', '1.'])
    # get the top k words from list of strings ignoring stopwords
    top_k_words = Counter([word for line in ans_human for word in line.split()
                           if word not in nltk_stopwords]).most_common(k)
    # top_k_words = Counter(" ".join(ans_pref_reason).split()).most_common(k)
    print(f"Top {k} words: {top_k_words}")
    return top_k_words


if __name__ == '__main__':
    project_path = "src/data/lfqa-biology-tud.zip"
    project = Project.from_zipped_xmi(project_path)
    print(project.annotators)
    print(project.custom_layers)
    print(project._annotation_info)
    layers = [layer for layer in project.custom_layers if not layer.__contains__("Answerpreference")]
    features = [project.features(layer)[-1] for layer in layers]

    for idx, (layer, feature) in enumerate(zip(layers, features)):
        type_name = layer.split(".")[-1]
        if type_name == "Completeness":
            type_name = "incomplete_ques"
        elif type_name == "Hardtounderstand":
            type_name = "hard"
        elif type_name == "Factuality":
            type_name = "factuality"
        elif type_name == "References":
            type_name = "reference_example"
        elif type_name == "Questionmisconception":
            type_name = "ques_misconception"
        elif type_name == "Irrelevant":
            type_name = "irrelevance"
        elif type_name == "CompletenessAnswer":
            type_name = "incomplete_ans"

        df = get_annotations(layer, feature, type_name)
        df.to_csv(f'./src/data/prolific/pilot_results_bio_tud/lfqa_pilot_{type_name}.csv', sep='\t')

    # path = "data/prolific/pilot_results_bio_v0/lfqa_pilot_complete.csv"
    # # layer_wise_analysis(path, "reference_example")
    # get_top_k_words(path, "ans_preference", 10)
