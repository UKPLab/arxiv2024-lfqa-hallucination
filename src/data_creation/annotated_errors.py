import pandas as pd
import ast
from utils import correct_text
import re
import json
from typing import List

from src.data_creation import utils

TASK_INSTRUCTION = {
    "irrelevance": "When given an instruction and evidence, evaluate whether the information in the evidence is "
                   "relevant to the instruction and provides valuable information for a meaningful response.\n"
                   "Use the '[Irrelevant]' and '[/Irrelevant]' tags to indicate irrelevance, and "
                   "'[Relevant]' tag to indicate relevance and usefulness.",

    "ans_preference": "When given an instruction and two corresponding pieces of evidence, evaluate which evidence "
                      "provides more factual, complete, and relevant information for a meaningful response.\n"
                      "Use the format '[Evidence1]' and '[Evidence2]' to indicate the informative evidence.",

    "factuality": "When given an instruction and evidence, evaluate whether the information in the evidence is "
                  "factually correct.\nUse the '[InspectFact]' and '[/InspectFact]' tags to indicate incorrect " 
                  "information. Please provide reasons for the indicated factual inconsistency in the format "
                  "'Reasons: [reason1, reason2,...]'.",
}


class ErrorDataset:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        """
        Load annotated data
        :return:
        """
        df = pd.read_csv(
            self.data_path,
            na_values=['NA', 'NaN', '', 'NULL', 'missing', "[]"],
            delimiter="\t",
        )
        df.reset_index(drop=True, inplace=True)
        return df

    def _get_correct_answers(self, example, answer_choice):
        """
        Get the error-free answers for the given aspect, answer choice (human or model)
        :param example:
        :param answer_choice:
        :param aspect:
        :return:
        """
        if example["ans1_label"].__contains__(answer_choice):
            answer = example["ans1_text"]

        elif example["ans2_label"].__contains__(answer_choice):
            answer = example["ans2_text"]
        else:
            return None
        return answer

    def _get_answer_specific_data(
            self,
            example: pd.Series,
            answer_choice: str,
            aspect: str,
            use_reason: bool = False
    ) -> (str, List, List):
        """
        Get the answer specific data for the given aspect, answer choice (human or model)
        :param example:
        :param answer_choice:
        :param aspect:
        :return:
        """
        retrieved_reasons = None
        if example["ans1_label"].__contains__(answer_choice) \
                and "answer1" in ast.literal_eval(example[f"{aspect}_label"]):
            answer = example["ans1_text"]
            aspect_labels = ast.literal_eval(example[f"{aspect}_label"])
            indexes = [idx for idx, value in enumerate(aspect_labels) if value == "answer1"]
            all_spans = ast.literal_eval(example[f"{aspect}_span"])
            retrieved_spans = [all_spans[idx] for idx in indexes]
            if use_reason:
                reasons = ast.literal_eval(example[f"{aspect}_reason"])
                retrieved_reasons = [reasons[idx] for idx in indexes]
        elif example["ans2_label"].__contains__(answer_choice) \
                and "answer2" in ast.literal_eval(example[f"{aspect}_label"]):
            answer = example["ans2_text"]
            aspect_labels = ast.literal_eval(example[f"{aspect}_label"])
            indexes = [idx for idx, value in enumerate(aspect_labels) if value == "answer2"]
            all_spans = ast.literal_eval(example[f"{aspect}_span"])
            retrieved_spans = [all_spans[idx] for idx in indexes]
            if use_reason:
                reasons = ast.literal_eval(example[f"{aspect}_reason"])
                retrieved_reasons = [reasons[idx] for idx in indexes]
        else:
            return None, None, None
        return answer, retrieved_spans, retrieved_reasons

    def create_tags(
            self,
            aspects: List,
            answer_choice: str,
            add_score: bool = False,
            use_reason: bool = False,
            use_all_data: bool = False,
            max_correct_answers: int = 10
    ):
        """
        Create the highlighted identifier tags for the given aspects
        :param aspects:
        :param answer_choice:
        :param add_score:
        :return:
        """
        df = self.load_data()
        aspect_tag: dict = {
            "factuality": "InspectFact",
            "irrelevance": "Irrelevant",
            "incomplete_ans": "InspectComp",
            "reference_example": "InspectUtil"
        }

        count = 0
        correct_answers_count = 0
        for i, ex in df.iterrows():
            output = ""
            # print(ex["source_file"])
            # if ex["source_file"] == "53b38757-cae0-4a27-8fd5-45c20b852c38.txt":
            #     # continue
            for aspect in aspects:
                tag = aspect_tag[aspect]
                if pd.isna(ex[f"{aspect}_label"]):
                    if not use_all_data:
                        continue
                    if correct_answers_count >= max_correct_answers:
                        continue
                    answer = self._get_correct_answers(ex, answer_choice)
                    reason = "The evidence is relevant to the instruction and provides valuable " \
                             "information for a meaningful response."
                    if answer is None:
                        continue
                    output = f"[Relevant]\nReasons: [{reason}]"
                    correct_answers_count += 1
                else:
                    # print(ex)
                    answer, spans, reasons = self._get_answer_specific_data(ex, answer_choice, aspect, use_reason)
                    print("ans:", answer)
                    print("span:", spans)
                    if answer is None:
                        continue
                    # print(answer)
                    # print(span)

                    for span in spans:
                        if output != "":
                            answer = output
                        start_index = answer.lower().find(correct_text(span).lower())
                        if start_index != -1:
                            # Find the start of the sentence
                            sentence_start = answer.rfind('.', 0, start_index) + 1
                            # Find the end of the sentence
                            end_index = answer.find('.', start_index) + 1 if answer.find('.', start_index) != -1 else len(answer)
                            # Extract the sentence containing the phrase
                            sentence = answer[sentence_start:end_index].strip()
                            # Wrap the found sentence in <p> tags
                            output = answer.replace(sentence, f'[{tag}]{sentence}[/{tag}]')
                            count += 1
                        elif span.__contains__("ANSWER"):
                            # wrap entire text in tags
                            output = f'[{tag}]{answer}[/{tag}]'
                        else:
                            print(answer)
                            print(span)
                            print("Phrase not found in the text.")
                            continue

                # check if open and close tags come together and remove them
                if output.__contains__(f"[{tag}][/{tag}]"):
                    output = output.replace(f"[{tag}][/{tag}]", "")

                # add aspect score to the highlighted text
                if add_score:
                    if aspect == "irrelevance":
                        name = "relevance"
                    else:
                        name = aspect
                    aspect_score = ex[f"{name}_{answer_choice}_score"]
                    output = f'{output} <Score={str(aspect_score)}>'

                # add reasons to the highlighted text
                if use_reason:
                    if reasons is not None:
                        output = f'{output}\nReasons: {reasons}'
                    else:
                        output = f'{output}\nReasons: []'
                print(output)
                # add highlighted text to df
                df.loc[i, f"{aspect}_highlighted"] = output
                df.loc[i, "identified_answer"] = answer

        # remove nan rows of highlighted text
        cols = [f"{aspect}_highlighted" for aspect in aspects]
        df.dropna(subset=cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # filter columns
        final_cols = ["question_text", "identified_answer"] + cols
        df = df[final_cols]
        # shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df


if __name__ == '__main__':
    data_path = "src/data/annotated_data/complete_data_scores.csv"
    dataset = ErrorDataset(data_path)
    aspects = ["factuality"]  # "irrelevance", "incomplete_ans", "reference_example"
    use_all_data = False
    use_reason = True
    add_score = False
    max_correct_answers = 10

    # write the question answers to a json file
    examples = []
    for aspect in aspects:
        if aspect != "ans_preference":
            complete_df = pd.DataFrame()
            for ans_choice in ["model", "human"]:
                df = dataset.create_tags(
                    aspects=aspects,
                    answer_choice=ans_choice,
                    use_reason=use_reason,
                    use_all_data=use_all_data,
                    add_score=add_score,
                    max_correct_answers=max_correct_answers
                )

                # concatenate the dfs for different answer choices
                if complete_df.empty:
                    complete_df = df
                else:
                    complete_df = pd.concat([complete_df, df], axis=0, ignore_index=True)
            complete_df = complete_df.loc[:, ~complete_df.columns.str.contains('^Unnamed')]
            for i, row in complete_df.iterrows():
                examples.append({
                    "instruction": f"{TASK_INSTRUCTION[aspect]}",
                    "input": f"Task instruction: {row['question_text']}\nEvidence: {row['identified_answer']}",
                    "output": row[f"{aspect}_highlighted"],
                })
        else:
            df = dataset.load_data()
            for i, row in df.iterrows():
                preference = re.findall(r'\d+', row[aspect])[0]
                output = f"[Evidence{preference}]"
                examples.append({
                    "instruction": f"{TASK_INSTRUCTION[aspect]}",
                    "input": f"Task instruction: {row['question_text']}\nEvidence1: {row['ans1_text']}\nEvidence2: {row['ans2_text']}",
                    "output": output,
                })
        utils.jdump(examples, f"src/data/annotated_data/{aspect}_detection.jsonl")
