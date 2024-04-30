import os.path

from inceptalytics import Project
from inceptalytics.utils import annotation_info_from_xmi_zip, SENTENCE_TYPE_NAME
import pandas as pd
from collections import Counter, OrderedDict
import ast
import re
import json
import numpy as np
from typing import List
from krippendorff import alpha

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class Annotation:
    def __init__(
            self,
            project_path: str,
            metadata_path: str,
            save_path: str,
            annotator_idx: List,
            include_pilot: bool = False
    ):
        self.project = Project.from_zipped_xmi(project_path, mode="")
        self.annotations = annotation_info_from_xmi_zip(project_path, mode="")
        self.metadata_path = metadata_path
        self.save_path = save_path
        self.annotator_idx = annotator_idx
        self.include_pilot = include_pilot

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
        # print(annotators)
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
        # select reduced view
        reduced_annos = self.project.select(
            annotation=feature_path,
            annotators=self.annotator_idx,
            source_files=source_files
        )

        df = reduced_annos.data_frame
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        # print(df.head)
        # print(df[f"{type_name}_annotation"][0])
        # remove the sentence column
        df.drop(columns=['sentence'], inplace=True)
        # change column names for readability
        for col in df.columns:
            # keep the original column name for later merge
            if col.startswith('source_file') or col.startswith('annotator'):
                continue
            elif col.startswith('_annotation'):
                df = df.rename(columns={col: col.replace('_annotation', f'{type_name}_annotation')})
            elif col.startswith('annotation'):
                df = df.rename(columns={col: col.replace('annotation', f'{type_name}_reason')})
            elif col.startswith('text'):
                df = df.rename(columns={col: col.replace('text', f'{type_name}_span')})
            else:
                df = df.rename(columns={col: col.replace(col, f'{type_name}_{col}')})

        # print(df.columns)
        # print(df.head())
        # print(df[f"{type_name}_annotation"][0])

        if "Completeness" in type_name:
            df[f'{type_name}_annotation'] = df[f'{type_name}_annotation'].apply(lambda x: x['Iscomplete'])
        else:
            df[f'{type_name}_annotation'] = df[f'{type_name}_annotation'].apply(lambda x: x['Preference'])
        # convert span, reason, begin, end, sentence text to list of strings
        df[f'{type_name}_span'] = df[f'{type_name}_span'].apply(lambda x: [x])
        df[f'{type_name}_reason'] = df[f'{type_name}_reason'].apply(lambda x: [x])
        df[f'{type_name}_begin'] = df[f'{type_name}_begin'].apply(lambda x: [x])
        df[f'{type_name}_end'] = df[f'{type_name}_end'].apply(lambda x: [x])
        df[f'{type_name}__sentence_text'] = df[f'{type_name}__sentence_text'].apply(lambda x: [x])

        df = df.groupby(['source_file', 'annotator']).agg(
            {f'{type_name}_span': 'sum', f'{type_name}_reason': 'sum', f'{type_name}_begin': 'sum',
             f'{type_name}_annotation': lambda x: x.tolist(), f'{type_name}_end': 'sum',
             f'{type_name}__sentence_text': 'sum'}).reset_index()
        # print(df.head)

        return df

    def collate_metadata(self, annotations_df):
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

        # remove numbering to match with metadata
        annotations_df["source_file"] = annotations_df["source_file"].apply(lambda x: x.split("_")[-1])
        # print(annotations_df.source_file.values)
        # filter meta df
        meta_df = meta_df[meta_df['source_file'].isin(annotations_df.source_file.values)]
        results_df = annotations_df.merge(meta_df, on="source_file")

        # print(results_df.columns)
        # print(results_df.head())

        return results_df

    def main(self):
        layers = [layer for layer in self.project.custom_layers]  # if not layer.__contains__("Answerpreference")]
        features = [self.project.features(layer)[-1] for layer in layers]

        # choose user preference
        def filter_true_answers(row, answers_col, boolean_col):
            answers = row[answers_col]
            boolean_values = row[boolean_col]
            return [answer for answer, boolean in zip(answers, boolean_values) if boolean]

        df = pd.DataFrame()
        for idx, (layer, feature) in enumerate(zip(layers, features)):
            type_name = layer.split(".")[-1]
            print(f"Processing layer: {type_name}")

            feature = self.project.features(layer)
            print(feature)
            data = self.get_layer_annotations(layer, feature, type_name)
            print(data.shape)

            data["source_file"] = data["source_file"].apply(lambda x: x.split("_")[-1])
            data[f"{type_name}_choice"] = data.apply(
                filter_true_answers, axis=1, args=(f"{type_name}_span", f"{type_name}_annotation")
            )
            # print(data.columns)
            # if dataframe is empty, then add the data
            if df.empty:
                df = data
            else:
                df = df.merge(data, on=["source_file", "annotator"], how="outer")
        # print(df.head)
        # remove duplicate rows based on source_file and annotator
        df = df.drop_duplicates(subset=["source_file", "annotator"]).reset_index(drop=True)
        df = df.fillna("")

        # remove unnecessary columns that contain the following strings
        remove_cols = ["_begin", "_end", "_sentence_text", "_sentence", "_reason", "_span", "_annotation"]
        df = df[[col for col in df.columns if not any(x in col for x in remove_cols)]]

        labelled_df = self.collate_metadata(df)

        def replace_answers(lst, ans1_label, ans2_label):
            return [ans1_label
                    if answer.__contains__('ANSWER1')
                    else ans2_label if answer.__contains__('ANSWER2') else answer
                    for answer in lst]

        labelled_df['AnswerPreference_choice'] = labelled_df.apply(
            lambda row: replace_answers(row['AnswerPreference_choice'], row['ans1_label'], row['ans2_label']), axis=1)
        labelled_df['Completeness_choice'] = labelled_df.apply(
            lambda row: replace_answers(row['Completeness_choice'], row['ans1_label'], row['ans2_label']), axis=1)

        # remove answer from the completion choice
        #############################################
        # not required in the case of held out data
        #############################################
        labelled_df['Completeness_choice'] = labelled_df['Completeness_choice'].apply(
            lambda x: [answer for answer in x if not answer == 'answer']
        )

        # remove unnecessary columns for ans labels
        labelled_df = labelled_df.drop(columns=['ans1_label', 'ans2_label'])

        labelled_df = labelled_df.groupby(['source_file']).agg(
            {
                "Completeness_choice": lambda x: x.tolist(),
                "AnswerPreference_choice": 'sum'
            }).reset_index()

        pd.set_option('display.max_rows', None)
        # print(labelled_df.columns)
        # print(labelled_df.head)
        #
        # print(labelled_df['AnswerPreference_choice'].value_counts())
        # print(labelled_df['Completeness_choice'].value_counts())
        return labelled_df


if __name__ == '__main__':

    category = "completeness"
    version = "eli5"
    dataset = "eli5"
    # check file path
    base_path = f"src/data/annotated_data/{category}"

    annotator_data = [
        {
            "subject": "held_out",
            "data": {
                "Manika": "KYcF8NuHe1VTHZzszNn3Sg",
                "650c3ea3bb7f3b500e2db7a1": "YBNCazbDTrgbIGpQHa2Hqw",
                "637fbedd06b96e980ee37a6d": "TSBjxfK9NgO-l1CSfbfkMQ",
            },
        },
        {
            "subject": "held_out_2",
            "data": {
                "Manika": "DEmoggkPJjEcNaEXtTJ1Qg",
                "650c3ea3bb7f3b500e2db7a1": "MAhQzH1ZaGy1ARosM0Ndcg",
                "637fbedd06b96e980ee37a6d": "Mxr7TTFLJkbQiBIOOaoMuA",
            },
        },
        {
            "subject": "asqa",
            "data": {
                "Manika": "vwPJMNt8Ad1zSWc9JOnd3Q",
                "650c3ea3bb7f3b500e2db7a1": "uw6tupn7gvi-p3H7pxtLfQ",
                "637fbedd06b96e980ee37a6d": "p-br2ARL0eIZD2-OxMjm1A",
            },
        },
        {
            "subject": "eli5",
            "data": {
                "Manika": "pP2ItEvh3O3ivvOOu7AjyQ",
                "650c3ea3bb7f3b500e2db7a1": "IpM93i1W8w9atvkjEXl2dA",
                "637fbedd06b96e980ee37a6d": "aIUh4M0tSjWJj-fPaHYLlA",
            },
        }
    ]

    for annotator in annotator_data:
        if annotator["subject"] == dataset:
            annotator_idx = list(annotator["data"].values())
            # for prolific_id, inception_id in annotator["data"].items():
            annotate = Annotation(
                project_path=f"{base_path}/qa-eval_{dataset}.zip",
                metadata_path=f"results/13b/llama2/{version}/",
                save_path=f"{base_path}/results_{category}_tud_{dataset}",
                annotator_idx=annotator_idx,
            )
            df = annotate.main()

    #             if isinstance(num_annotator, int):
    #                 num_annotator += 1
    #
    # path = "data/prolific/pilot_results_bio_v0/lfqa_pilot_complete.csv"
    # print(df['AnswerPreference_choice'].value_counts())
    # print(df.head())
    # print(df['Completeness_choice'].value_counts())
    # print(df['AnswerPreference_choice'].values)

    values = df['AnswerPreference_choice'].values
    # response_array = np.array(values)
    # transposed_array = list(map(list, zip(*response_array)))
    # print(transposed_array)
    # alpha_value = alpha(reliability_data=transposed_array)
    # print("Krippendorff's alpha:", alpha_value)


    rater1, rater2, rater3 = [], [], []
    for value in values:
        # print(value)
        rater1.append(value[0])
        rater2.append(value[1])
        rater3.append(value[2])

    # print(rater1)
    ratings1 = [1 if x=="refined_answer" else 0 for x in rater1]
    ratings2 = [1 if x=="refined_answer" else 0 for x in rater2]
    ratings3 = [1 if x=="refined_answer" else 0 for x in rater3]

    response_array = np.transpose([ratings1, ratings2, ratings3])
    # data = list(map(list, zip(*response_array)))

    data = list(response_array)
    print(data)

    repr_data = []
    for rate in data:
        rate = list(rate)
        # print(Counter(rate))
        repr_data.append([rate.count(0), rate.count(1)])
    print(repr_data)


    # from krippendorff import alpha
    alpha_value = alpha(reliability_data=repr_data, level_of_measurement='nominal')
    print("Krippendorff's alpha:", alpha_value)


    # convert list of lists to list of strings
    # rater1 = [ for x in rater1]
    # print(rater1)





