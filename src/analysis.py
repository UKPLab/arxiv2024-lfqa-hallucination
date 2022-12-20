from inceptalytics import Project
from inceptalytics.utils import annotation_info_from_xmi_zip, SENTENCE_TYPE_NAME
import pandas as pd


def get_annotations(layer=None, feature=None, type_name=None):
    if feature:
        feature_path = f'{layer}>{feature}'
    else:
        feature_path = f'{layer}'

    source_files = ["00f64ae1-c96a-4b2e-9912-c40e295e2565.txt", "02062ffd-f0bb-4566-847c-fc7057d4ce0b.txt",
                    "02408398-02c5-4afc-866d-4225e869d49a.txt", "0242d81a-2c1b-4021-9aa1-c5cfcf38a784.txt",
                    "031d790a-7d9d-498e-a36d-a28eaca304ac.txt", "03235d4c-dcb0-4cac-8dc2-5d17b072f5ac.txt",
                    "037f220f-7797-4b35-bcf0-201336c4127b.txt", "03804547-6f04-472b-82fc-bf073985dfc2.txt",
                    "03fd2c88-e3e9-4ba5-8641-fa7814df6b08.txt", "04ac3833-c4d7-4bcb-a3f4-f133819ef4df.txt",
                    "059fa085-f357-478a-83c1-fc5933bbad6a.txt", "05b4333c-1c42-4ede-b62c-830c38cbfea2.txt",
                    "06210903-a207-4171-901b-45d89165bec3.txt", "0725d873-e32c-4c8e-bbcf-8e3eb6a8a0cd.txt",
                    "075398ee-9f9f-4ff7-97aa-7d0315844130.txt"]
    annotators = ['rachneet', 'yixiao']

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


if __name__ == '__main__':
    project_path = "data/lfqa-pilot-v2.zip"
    project = Project.from_zipped_xmi(project_path)
    layers = [layer for layer in project.layers if layer.startswith("webanno")
              and not layer.__contains__("Answerpreference")]
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
        df.to_csv(f'data/pilot_results_v4/lfqa_pilot_{type_name}.csv', sep='\t')
