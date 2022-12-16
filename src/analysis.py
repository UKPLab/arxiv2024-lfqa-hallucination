from inceptalytics import Project
from inceptalytics.utils import annotation_info_from_xmi_zip, SENTENCE_TYPE_NAME
import pandas as pd


def get_annotations(file_path, layer, feature=None):
    # load project
    project = Project.from_zipped_xmi(file_path)
    print('No. of files: ', len(project.source_file_names))
    print('Layers:', project.layers)
    print(f'Features: {project.features(_layer)}')

    if feature:
        feature_path = f'{_layer}>{feature}'
    else:
        feature_path = f'{_layer}'

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

    print('# pref. annotations in view:', reduced_annos.count())
    print('# annotations per file per annotator', reduced_annos.count(grouped_by=['source_file', 'annotator']))

    df = reduced_annos.data_frame
    df.head()
    return df


if __name__ == '__main__':
    _layer = "webanno.custom.Hardtounderstand"
    feature = "Reasonfordifficulty"
    df = get_annotations("data/lfqa-pilot-v2.zip", _layer, feature=feature)
    print(df.head())
    df.to_csv("./data/pilot_results_v2/lfqa-pilot-hard.csv", sep="\t")
