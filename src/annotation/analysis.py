import os.path

from inceptalytics import Project
from inceptalytics.utils import annotation_info_from_xmi_zip, SENTENCE_TYPE_NAME
import pandas as pd
from collections import Counter, OrderedDict
import ast
import re
import json
import numpy as np

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class Annotation:
    def __init__(self, project_path, metadata_path, save_path):
        self.project = Project.from_zipped_xmi(project_path)
        self.annotations = annotation_info_from_xmi_zip(project_path)
        self.metadata_path = metadata_path
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def answer_annotations(self, layer, feature, files, annotators):
        """
        Get answer preference annotation from the inception project
        :param layer:
        :param feature:
        :param files:
        :param annotators:
        :param annotations:
        :return:
        """

        project = {
            "layer": layer,
            "features": feature,
            "annotations": self.annotations,
            "source_files": files,
            "annotators": annotators
        }
        annotators = project["annotators"]
        source_files = project["source_files"]
        features = project["features"]

        _annotation_info = pd.DataFrame(project["annotations"], columns=['cas', 'source_file', 'annotator'])
        # filter
        df = _annotation_info.query('annotator == @annotators')
        df = df.query('source_file == @source_files')
        entries: list = []
        for cas, source_file, annotator in df.itertuples(index=False, name=None):
            try:
                for annotation in cas.select(project["layer"]):
                    # print(annotation)
                    entry = (
                         annotation[features[0]],
                         annotation[features[1]],
                         source_file,
                         annotator
                    )
                    entries.append(entry)
            except Exception as e:
                print(e.__traceback__)

        columns = [features[0].lower(), features[1].lower(), 'source_file', 'annotator']
        index = ['source_file', 'annotator']
        annotations = pd.DataFrame(entries, columns=columns).set_index(index)
        # change column names
        if 'reason' in annotations.columns:
            annotations = annotations.rename(columns={'reason': 'ans_preference_reason'})
        if 'preference' in annotations.columns:
            annotations = annotations.rename(columns={'preference': 'ans_preference'})
        return annotations

    def get_layer_annotations(self, layer=None, feature=None, type_name=None):
        """
        Get layer annotations from the inception project
        :param layer:
        :param feature:
        :param type_name:
        :param annotations:
        :return:
        """

        if feature:
            feature_path = f'{layer}>{feature}'
        else:
            feature_path = f'{layer}'
        source_files = [file for file in self.project.source_file_names if not file.__contains__("completion.txt")]
        # print(f"Source files: {source_files}")
        # biology
        # annotators = ['ALDTHTi6bdUV_PxRqESvew', 'ElEWCTeqc0gwYO7Ze9yW0Q', 'frFLqX9g5biFUM6tBMAoFA']
        annotators = ['kx7IZQV_R_uhMxYEyuv-8w', '4E4KDglOHRxflovIhgyahA', '9qDy2dkwY13nFOeb_9EFwg']
        # economics
        # annotators = ['U4U-HsGbDSjUjGsQYqwzCA', 'uCKTljvJIvpxenurEySxgA', 'Qp4YXOnvjo3VVadyDJRMAA']

        # select reduced view
        reduced_annos = self.project.select(
            annotation=feature_path,
            annotators=annotators,
            source_files=source_files
        )

        if type_name == "answer_preference":
            return self.answer_annotations(
                layer=layer,
                feature=feature,
                annotators=annotators,
                files=source_files,
            )

        df = reduced_annos.data_frame
        # print(df.columns)

        if "reference" in type_name:
            # get the reference helpfulness
            df['helpful'] = df['_annotation'].apply(lambda x: x['Isithelpful'])

        # remove the annotation column and sentence column
        df.drop(columns=['sentence', '_annotation'], inplace=True)
        # change column names for readability
        for col in df.columns:
            # keep the original column name for later merge
            if col.startswith('source_file') or col.startswith('annotator'):
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

    def collate_metadata(self):
        """
        Collates metadata to the collected annotations
        Matches the randomized human model answer choices to the annotator selection in a neat dataframe
        :param path: the path to the annotations
        :return:
        """

        metadata = os.path.join(self.metadata_path, "metadata.json")
        with open(metadata, "r") as file:
            meta = json.load(file)

        struct_meta = [dict(v, id=k) for x in meta for k, v in x.items()]
        meta_df = pd.DataFrame(struct_meta)
        meta_df.columns = ["ans1_label", "ans2_label", "source_file"]
        meta_df["source_file"] = meta_df["source_file"] + ".txt"
        annotations = os.path.join( self.save_path, "lfqa_pilot_answer_preference.csv")
        annotations_df = pd.read_csv(annotations, sep="\t")
        original_source_files = annotations_df.source_file.values
        # remove numbering to match with metadata
        annotations_df["source_file"] = annotations_df["source_file"].apply(lambda x: x.split("_")[-1])
        # print(annotations_df.source_file.values)
        # filter meta df
        meta_df = meta_df[meta_df['source_file'].isin(annotations_df.source_file.values)]
        results_df = annotations_df.merge(meta_df, on="source_file")

        # get the source files
        source_files = results_df["source_file"].unique()
        original_data = {"question_text": [], "ans1_text": [], "ans2_text": [], "ques_start": [], "ans_start": []}

        # use original source ids here
        for file in original_source_files:
            # open the file
            with open(self.metadata_path+file, "r") as f:
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
        results_df.to_csv(os.path.join(self.save_path, "lfqa_pilot_answer_preference_with_labels.csv"), sep="\t")

    def collate_annotations(self):
        """Combines all the annotations from the different annotators"""

        # get all files from the directory
        files = [f for f in os.listdir(self.save_path) if os.path.isfile(os.path.join(self.save_path, f)) and
                 not f.endswith("answer_preference.csv")]
        # read all the files and combine them into one dataframe
        df = pd.DataFrame()
        for file in files:
            data = pd.read_csv(os.path.join(self.save_path, file), sep="\t", index_col=0)
            data["source_file"] = data["source_file"].apply(lambda x: x.split("_")[-1])
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
        # df["incomplete_ques_label"] = df.apply(_feature_labels, axis=1, args=("incomplete_ques_end", "question"))
        df["incomplete_ans_label"] = df.apply(_feature_labels, axis=1, args=("incomplete_ans_end", "answer"))
        df["reference_example_label"] = df.apply(_feature_labels, axis=1, args=("reference_example_end", "answer"))
        # df["hard_label"] = df.apply(_feature_labels, axis=1, args=("hard_end", "answer"))

        # remove unnecessary columns that contain the following strings
        remove_cols = ["_begin", "_end", "_sentence_text", "_sentence", "_annotation"]
        df = df[[col for col in df.columns if not any(x in col for x in remove_cols)]]

        # get index of  question_text and ans1_text and ans2_text columns
        question_index = df.columns.get_loc("question_text")
        ans1_index = df.columns.get_loc("ans1_text")
        ans2_index = df.columns.get_loc("ans2_text")
        cols = df.columns.tolist()

        # move the question_text and ans1_text and ans2_text columns after the source_file and annotator columns
        cols = cols[:2] + cols[question_index:ans2_index + 1] + cols[2:question_index] + cols[ans2_index + 1:]
        df = df[cols]

        # fill na values with empty list
        df = df.fillna("[]")
        # save the dataframe to a csv file
        df.to_csv(os.path.join(self.save_path, "lfqa_pilot_complete.csv"), sep="\t")

    def main(self):
        layers = [layer for layer in self.project.custom_layers]  # if not layer.__contains__("Answerpreference")]
        features = [self.project.features(layer)[-1] for layer in layers]
        for idx, (layer, feature) in enumerate(zip(layers, features)):
            type_name = layer.split(".")[-1]

            if type_name == "Factuality":
                type_name = "factuality"
            elif type_name == "References":
                type_name = "reference_example"
            elif type_name == "Questionmisconception":
                type_name = "ques_misconception"
            elif type_name == "Irrelevant":
                type_name = "irrelevance"
            elif type_name == "CompletenessAnswer":
                type_name = "incomplete_ans"
            elif type_name == "Answerpreferencev1":
                type_name = "answer_preference"

                feature = self.project.features(layer)

            df = self.get_layer_annotations(layer, feature, type_name)
            df.to_csv(os.path.join(self.save_path, f"lfqa_pilot_{type_name}.csv"), sep="\t")
        print("---- Saving layer annotations: DONE ----")
        # add labels to randomized answer preferences from the metadata
        self.collate_metadata()
        print("---- Mapped metadata to annotations: DONE ----")
        # finally, collate all annotations
        self.collate_annotations()
        print("---- Collating all annotations: DONE ----")


def layer_wise_analysis(path, layer):
    df = pd.read_csv(path, sep='\t')
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


def get_top_k_words(path, layer, k):

    """
    Function to get the top k words in the answer_preference_reason in df
    :param path:
    :param layer:
    :param k:
    :return:
    """

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
    num_annotator = 3  # 1, 2 or 3
    category = "biology"
    annotate = Annotation(
        project_path=f"src/data/projects/{category}/lfqa-{category}-tud-{num_annotator}.zip",
        metadata_path=f"src/data/human_annotations/gpt4/{category}/zero/{category}/v0/",
        save_path=f"src/data/prolific/results_{category}_tud_{num_annotator}"
    )
    annotate.main()

    # path = "data/prolific/pilot_results_bio_v0/lfqa_pilot_complete.csv"
    # # layer_wise_analysis(path, "reference_example")
    # get_top_k_words(path, "ans_preference", 10)
