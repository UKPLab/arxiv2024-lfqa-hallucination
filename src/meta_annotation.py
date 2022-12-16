from inceptalytics import Project
from inceptalytics.utils import annotation_info_from_xmi_zip, SENTENCE_TYPE_NAME
from cassis import View, Cas
import pandas as pd


def _load_project(file):
    project = Project.from_zipped_xmi(file)
    print('Layers:', project.layers)
    preference_layer = "webanno.custom.Answerpreferencev1"
    features = project.features(preference_layer)
    print(f'Features: {project.features(preference_layer)}')
    print('Annotators:', project.annotators)
    feature = 'Reason'
    feature_path = f'{preference_layer}'

    source_files = ["00f64ae1-c96a-4b2e-9912-c40e295e2565.txt", "02062ffd-f0bb-4566-847c-fc7057d4ce0b.txt",
                    "02408398-02c5-4afc-866d-4225e869d49a.txt", "0242d81a-2c1b-4021-9aa1-c5cfcf38a784.txt",
                    "031d790a-7d9d-498e-a36d-a28eaca304ac.txt", "03235d4c-dcb0-4cac-8dc2-5d17b072f5ac.txt",
                    "037f220f-7797-4b35-bcf0-201336c4127b.txt", "03804547-6f04-472b-82fc-bf073985dfc2.txt",
                    "03fd2c88-e3e9-4ba5-8641-fa7814df6b08.txt", "04ac3833-c4d7-4bcb-a3f4-f133819ef4df.txt",
                    "059fa085-f357-478a-83c1-fc5933bbad6a.txt", "05b4333c-1c42-4ede-b62c-830c38cbfea2.txt",
                    "06210903-a207-4171-901b-45d89165bec3.txt", "0725d873-e32c-4c8e-bbcf-8e3eb6a8a0cd.txt",
                    "075398ee-9f9f-4ff7-97aa-7d0315844130.txt"]
    annotators = ['rachneet', 'yixiao']
    annotations = annotation_info_from_xmi_zip(file)

    return {"layer": preference_layer, "features": features, "annotations": annotations,
            "source_files": source_files, "annotators": annotators}


def annotations(file):

    project = _load_project(file)
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
    return annotations


def analyze_results(path):
    df = pd.read_csv(path+"lfqa-pilot-answer-preference-with-labels.csv", sep="\t", index_col=0)
    annotators = df.annotator.unique()

    annotator_1_pref = df.query('annotator == @annotators[0]').preference.values
    annotator_2_pref = df.query('annotator == @annotators[1]').preference.values
    # annotator 1 to binary
    annotator_1_pref = [1 if x == 'Answer 1' else 0 for x in annotator_1_pref]
    # annotator 2 to binary
    annotator_2_pref = [1 if x == 'Answer 1' else 0 for x in annotator_2_pref]
    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(annotator_1_pref, annotator_2_pref)
    print("The IAA is: ", kappa)

    df["true_answer"] = None
    df.loc[df["preference"] == "Answer 1", "true_answer"] = df["answer_1"]
    df.loc[df["preference"] == "Answer 2", "true_answer"] = df["answer_2"]

    num_annotations = int(df.shape[0]/len(annotators))
    human_selection = list(df.groupby(["annotator"])["true_answer"].apply(lambda x: x[x.str.contains('human')].count()).values)
    model_selection = [num_annotations-x for x in human_selection]

    # groupby source file and annotator and count the number of human selections
    # and list the annotators who selected human
    df["human_selection"] = df["true_answer"].str.contains("human")
    df["model_selection"] = ~df["true_answer"].str.contains("human")
    df["human_selection"] = df["human_selection"].astype(int)
    df["model_selection"] = df["model_selection"].astype(int)
    df["human_selection"] = df["human_selection"].astype(str)
    df["model_selection"] = df["model_selection"].astype(str)
    df["human_selection"] = df["human_selection"].str.replace("1", "human")
    df["model_selection"] = df["model_selection"].str.replace("1", "model")
    df["human_selection"] = df["human_selection"].str.replace("0", "")
    df["model_selection"] = df["model_selection"].str.replace("0", "")

    # group by source file and annotator and count the number of human selections
    # and list the annotators who selected human
    result = df.groupby(["source_file", "annotator"])["human_selection", "model_selection"].\
        agg(lambda x: x.value_counts().index[0])

    res = {}
    # create a new column per source file and count the number of human selections and
    # model selections from different annotators as a list
    for idx, (source_file, annotator) in enumerate(result.index):
        human_selection = result.loc[(source_file, annotator), "human_selection"]
        model_selection = result.loc[(source_file, annotator), "model_selection"]
        if human_selection == "human":
            if source_file not in res:
                res[source_file] = {"human": [annotator], "model": []}
            else:
                res[source_file]["human"].append(annotator)
        if model_selection == "model":
            if source_file not in res:
                res[source_file] = {"human": [], "model": [annotator]}
            else:
                res[source_file]["model"].append(annotator)

    new_df = pd.DataFrame(res).T
    new_df["agreement"] = None
    new_df.loc[new_df["human"].str.len() == annotators.shape[0], "agreement"] = "human"
    new_df.loc[new_df["model"].str.len() == annotators.shape[0], "agreement"] = "model"
    agreement = new_df["agreement"].value_counts()
    print(agreement)


if __name__ == '__main__':
    # file = "data/lfqa-pilot-v2.zip"
    # # get annotations
    # annotations = annotations(file)
    # annotations.to_csv("./data/pilot_results_v2/lfqa-pilot-answer-preference.csv", sep="\t", index=True)

    filepath = './data/pilot_results_v2/'
    analyze_results(filepath)
