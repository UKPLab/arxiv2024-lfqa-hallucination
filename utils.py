import os
import json
import random
import uuid
import re
import ast


import pandas as pd

def convert_excel_to_text(file_path, file_name):
    """Converts a tsv file to a text file"""

    df = pd.read_csv(file_path+file_name, sep='\t', encoding='utf-8', header=0)

    # read each row of the dataframe and write it to a text file
    documents = [f"{file_name.replace('.tsv', str('_')+str(i+1)+'.txt')}"
                 for i in range(len(df.index))]
    c = 0

    for index, row in df.iterrows():
        question = f"QUESTION:\n{row['question_text'].capitalize()}".replace('<br />', '\n').replace('\r', '')
        answer = f"ANSWER1:\n{row['answer1'].capitalize()}".replace('<br />', '\n').replace('\r', '')
        answer2 = f"ANSWER2:\n{row['answer2'].capitalize()}".replace('<br />', '\n').replace('\r', '')

        save_path = file_path+file_name.split('_')[0].lower()+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path+documents[index], 'w') as f:
            f.write(question+'\n\n'+answer+'\n\n'+question+'\n\n'+answer2)


def convert_jsonl_to_text(file_path, file_name, version):
    """
    Converts a jsonl file to a text file
    :param file_path:
    :param file_name:
    :return:
    """
    with open(file_path+file_name, 'r') as f:
        data = f.readlines()
    documents = [f"{file_name.replace('.jsonl', str('_')+str(i+1)+'.txt')}"
                 for i in range(len(data))]
    data_config = []
    for i in range(len(data)):
        example = json.loads(data[i])
        question = f"QUESTION:\n{example['question_text'].capitalize()}".replace('<br />', '\n').replace('\r', '')
        random_no = random.randint(1, 2)
        human_answer = f"ANSWER{random_no}:\n{example['human_answer'].capitalize()}".replace('<br />', '\n').replace(
            '\r', '')
        save_path = file_path + file_name.split('_')[1].lower() + "/" + file_name.split('_')[0].lower() + "/"
        # print(save_path)
        if version:
            save_path += version+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unique_id = str(uuid.uuid4())
        if random_no == 1:
            model_answer = f"ANSWER2:\n{example['model_answer'].capitalize()}".replace('<br />', '\n').replace('\r', '')
            with open(save_path + unique_id + '.txt', 'w') as f:
                f.write(question + '\n\n' + human_answer + '\n\n'+ question +'\n\n'+ model_answer)
            # save metadata
            metadata = {unique_id: {"Answer1": "human_answer", "Answer2": "model_answer"}}
        else:
            model_answer = f"ANSWER1:\n{example['model_answer'].capitalize()}".replace('<br />', '\n').replace('\r', '')
            with open(save_path + unique_id + '.txt', 'w') as f:
                f.write(question + '\n\n' + model_answer + '\n\n'+ question +'\n\n'+ human_answer)
            # save metadata
            metadata = {unique_id: {"Answer1": "model_answer", "Answer2": "human_answer"}}

        data_config.append(metadata)
        # break
    # print(data_config)
    with open(save_path + 'metadata.json', 'w') as f:
        json.dump(data_config, fp=f, indent=4)


def collate_metadata(data_path, save_path):
    """
    Collates metadata to the collected annotations
    :param path: the path to the annotations
    :return:
    """

    metadata = f"{data_path}metadata.json"
    with open(metadata, "r") as file:
        meta = json.load(file)

    struct_meta = [dict(v, id=k) for x in meta for k, v in x.items()]
    meta_df = pd.DataFrame(struct_meta)
    meta_df.columns = ["ans1_label", "ans2_label", "source_file"]
    meta_df["source_file"] = meta_df["source_file"] + ".txt"
    annotations = save_path + "lfqa_pilot_answer_preference.csv"
    annotations_df = pd.read_csv(annotations, sep="\t")

    results_df = pd.merge(annotations_df, meta_df, on="source_file")
    # get the source files
    source_files = results_df["source_file"].unique()

    original_data = {"question_text": [], "ans1_text": [], "ans2_text": [], "ques_start": [], "ans_start": []}
    for file in source_files:
        # open the file
        with open(data_path+file, "r") as f:
            # read data and remove new lines
            data = f.read()
            ques_starts = [m.start() for m in re.finditer("QUESTION:", data)]
            ans1_starts = [m.start() for m in re.finditer("ANSWER1:", data)]
            ans2_starts = [m.start() for m in re.finditer("ANSWER2:", data)]
            ans_starts = ans1_starts + ans2_starts
            data = data.replace("\n", "")
            # split by question and remove the empty element
            qa_data = data.split("QUESTION:")
            qa_data = [x for x in qa_data if x != ""]
            # qa1, qa2 = data.split("QUESTION:")[1:]
            question, ans1 = qa_data[0].split("ANSWER1:")
            _, ans2 = qa_data[1].split("ANSWER2:")

            original_data["question_text"].append(question)
            original_data["ans1_text"].append(ans1)
            original_data["ans2_text"].append(ans2)
            original_data["ques_start"].append(ques_starts)
            original_data["ans_start"].append(ans_starts)

    # create a dataframe with source files and the original data
    original_df = pd.DataFrame(original_data)
    original_df["source_file"] = source_files

    # merge the original data with the results
    results_df = pd.merge(results_df, original_df, on="source_file")
    results_df.to_csv(save_path+"lfqa_pilot_answer_preference_with_labels.csv", sep="\t")


def collate_annotations(path):
    """Combines all the annotations from the different annotators"""
    # get all files from the directory
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # read all the files and combine them into one dataframe
    df = pd.DataFrame()
    for file in files:
        data = pd.read_csv(path+file, sep="\t", index_col=0)
        # print(data.columns)
        # if dataframe is empty, then add the data
        if df.empty:
            df = data
        # merge the data on source_file and annotator columns and put NaN if there is no value
        else:
            df = df.merge(data, on=["source_file", "annotator"], how="outer")
        # df = df.merge(data, on=["source_file", "annotator"])
    # remove duplicate rows based on source_file and annotator
    df = df.drop_duplicates(subset=["source_file", "annotator"]).reset_index(drop=True)
    # fill na values with null string
    df = df.fillna("")

    def _feature_labels(row, feature_name, label_name):
        labels = []
        if row[feature_name] != "":
            for idx, pos in enumerate(ast.literal_eval(row[feature_name])):
                if pos < ast.literal_eval(row["ques_start"])[1]:
                    labels.append(f"{label_name}1")
                else:
                    labels.append(f"{label_name}2")
        return labels

    # create new columns with the tags
    df["ques_misconception_label"] = df.apply(_feature_labels, axis=1, args=("ques_misconception_end", "question"))
    df["factuality_label"] = df.apply(_feature_labels, axis=1, args=("factuality_end", "answer"))
    df["irrelevance_label"] = df.apply(_feature_labels, axis=1, args=("irrelevance_end", "answer"))
    df["incomplete_ques_label"] = df.apply(_feature_labels, axis=1, args=("incomplete_ques_end", "question"))
    df["incomplete_ans_label"] = df.apply(_feature_labels, axis=1, args=("incomplete_ans_end", "answer"))
    df["reference_example_label"] = df.apply(_feature_labels, axis=1, args=("reference_example_end", "answer"))
    df["hard_label"] = df.apply(_feature_labels, axis=1, args=("hard_end", "answer"))

    # remove unnecessary columns that contain the following strings
    remove_cols = ["_begin", "_end", "_sentence_text", "_sentence", "_annotation"]
    df = df[[col for col in df.columns if not any(x in col for x in remove_cols)]]

    # get index of  question_text and ans1_text and ans2_text columns
    question_index = df.columns.get_loc("question_text")
    ans1_index = df.columns.get_loc("ans1_text")
    ans2_index = df.columns.get_loc("ans2_text")
    cols = df.columns.tolist()

    # move the question_text and ans1_text and ans2_text columns after the source_file and annotator columns
    cols = cols[:2] + cols[question_index:ans2_index+1] + cols[2:question_index] + cols[ans2_index+1:]
    df = df[cols]

    # fill na values with empty list
    df = df.fillna("[]")

    # save the dataframe to a csv file
    df.to_csv(path+"lfqa_pilot_complete.csv", sep="\t")


if __name__ == '__main__':
    save_path = 'src/data/pilot_results_v4/'
    data_path = 'src/data/human_annotations/gpt3/biology/v1/'
    file_name = 'Biology_gpt3_2shot_knn_revised.jsonl'
    # convert_excel_to_text(filepath, file_name)
    # convert_jsonl_to_text(filepath, file_name, version='v2')
    # collate_metadata(data_path, save_path)
    collate_annotations(save_path)
