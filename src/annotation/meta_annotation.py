from inceptalytics import Project
from inceptalytics.utils import annotation_info_from_xmi_zip, SENTENCE_TYPE_NAME
from cassis import View, Cas
import pandas as pd


def analyze_results(path):
    df = pd.read_csv(path+"lfqa_answer_preference_with_labels.csv", sep="\t", index_col=0)
    annotators = df.annotator.unique()
    print(annotators)

    # annotator_1_pref = df.query('annotator == @annotators[0]').ans_preference.values
    # annotator_2_pref = df.query('annotator == @annotators[1]').ans_preference.values

    # # annotator 1 to binary
    # annotator_1_pref = [1 if x == 'Answer 1' else 0 for x in annotator_1_pref]
    # # annotator 2 to binary
    # annotator_2_pref = [1 if x == 'Answer 1' else 0 for x in annotator_2_pref]
    # from sklearn.metrics import cohen_kappa_score
    # kappa = cohen_kappa_score(annotator_1_pref, annotator_2_pref)
    # print("The IAA is: ", kappa)

    df["true_answer"] = None
    df.loc[df["ans_preference"] == "Answer 1", "true_answer"] = df["ans1_label"]
    df.loc[df["ans_preference"] == "Answer 2", "true_answer"] = df["ans2_label"]

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
    # analysis
    filepath = './src/data/prolific/pilot_results_bio_tud/'
    analyze_results(filepath)
