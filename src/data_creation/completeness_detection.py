import pandas as pd
import ast
from utils import correct_text
import re
import json
from typing import List

from src.data_creation import utils
# avoid future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

TASK_INSTRUCTION = {
    "irrelevance": "When given a question and answer statements, evaluate whether each given statement is relevant "
                   "for answering the question meaningfully and coherently follows the preceding statement. \n Use the "
                   "'[Irrelevant]' tag to indicate irrelevance, and '[Relevant]' tag to indicate relevance, "
                   "with reasons.\n Please note that the answer can have single, multiple, or no irrelevant statements.",

    "incomplete_ans": "When given a question and answer statements, evaluate whether each given statement provides "
                      "sufficient information for answering the question. \n Use the '[Incomplete]' tag to indicate "
                      "answer incompleteness, and '[Complete]' tag to indicate completeness, with reasons.\n Please "
                      "note that the answer can have single, multiple, or no incomplete statements.",
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
        :param example: example row
        :param answer_choice: randomized answer choice
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

    def _split_answer(self, answer):
        """
        Split the answer into sentences
        :param answer:
        :return:
        """
        # sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', answer)
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        # join answer with the sentence number
        sentences = [f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)]
        return sentences

    def create_tags(
            self,
            aspect: str,
            answer_choice: str,
            add_score: bool = False,
            use_reason: bool = False,
            use_all_data: bool = False,
            max_correct_answers: int = None
    ):
        """
        Create the highlighted identifier tags for the given aspects
        :param aspect:
        :param answer_choice:
        :param add_score:
        :return:
        """
        df = self.load_data()
        aspect_tag: dict = {
            "factuality": "InspectFact",
            "irrelevance": "Irrelevant",
            "incomplete_ans": "Incomplete",
            "reference_example": "InspectUtil"
        }

        count = 0
        correct_answers_count = 0
        rel_count = 0
        if max_correct_answers is None:
            max_correct_answers = df.shape[0]
        for i, ex in df.iterrows():
            output = ""
            tag = aspect_tag[aspect]
            # print(tag)
            if pd.isna(ex[f"{aspect}_label"]):
                if answer_choice == "human":
                    continue
                tag = "[Complete]"
                if not use_all_data:
                    continue
                if correct_answers_count >= max_correct_answers:
                    continue
                gold_answer = self._get_correct_answers(ex, "model")
                if gold_answer is None:
                    continue
                # print(gold_answer)
                sentences = re.split(r'(?<=[.!?])\s+', gold_answer)
                response = ""
                # print(sentences)
                for idx, sentence in enumerate(sentences):
                    response += f'{idx+1}. {tag}\n'
                response = response.strip()
                # print(response)
                # print(answer)
                # reason = "The statement is relevant to the instruction and provides valuable " \
                #          "information for a meaningful response."
                # if answer is None:
                #     continue
                # output = f"[Relevant]\nReasons: [{reason}]"
                correct_answers_count += 1
                # print(correct_answers_count)
            else:
                # print(i)
                # print(ex["question_text"])
                answer, spans, reasons = self._get_answer_specific_data(ex, answer_choice, aspect, use_reason)
                # remove new line characters and \r from the reasons
                if reasons is not None:
                    reasons = [reason.replace("\n", "") for reason in reasons]
                    reasons = [reason.replace("\r", "") for reason in reasons]

                    sep_spans = []
                    sep_reasons = []
                    for idx, span in enumerate(spans):
                        sents = re.split(r'(?<=[.!?])\s+', span)
                        sep_spans.extend(sents)
                        sep_reasons.extend([reasons[idx]] * len(sents))

                    if len(sep_spans) != len(sep_reasons):
                        continue

                if answer is None:
                    continue

                # print(answer)
                # print(spans)
                # print(reasons)
                # print(sep_spans)
                # print("*"*50)

                gold_answer = answer
                for idx, span in enumerate(sep_spans):
                    if span.rstrip(".").isdigit():
                        sep_reasons.pop(idx)
                        continue
                    if output != "":
                        answer = output
                    # print(span)
                    # print(answer)
                    # print(sentences)
                    start_index = answer.lower().find(correct_text(span).lower())
                    if start_index != -1:
                        # Find the start of the sentence
                        sentence_start = answer.rfind('.', 0, start_index) + 1
                        # Find the end of the sentence
                        end_index = answer.find('.', start_index) + 1 \
                            if answer.find('.', start_index) != -1 else len(answer)
                        # Extract the sentence containing the phrase
                        sentence = answer[sentence_start:end_index].strip()
                        # Wrap the found sentence in <p> tags
                        output = answer.replace(sentence, f'{tag}: {sentence}[/{tag}].')
                        count += 1
                    elif span.__contains__("ANSWER"):
                        # add tag at the beginning of each sentence
                        sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', answer)
                        sep_reasons = sep_reasons * len(sentences)
                        # output = ""
                        for sentence in sentences:
                            output = f'{output} {tag}: {sentence}[/{tag}].'
                    else:
                        # print(answer)
                        # print(span)
                        print("Phrase not found in the text.")
                        continue

                # check if open and close tags come together and remove them
                if output.__contains__(f"[/{tag}]. {tag}"):
                    output = output.replace(f"[/{tag}]. {tag}:", f" {tag}:")

                response = ""
                sentences = self._split_answer(output)
                relevance_reason = "The statement is relevant to the question and provides valuable " \
                                   "information for a meaningful response."
                for idx, sentence in enumerate(sentences):
                    # print(sentence[2:])
                    if sentence[3:].lstrip().startswith(f"{tag}"):

                        if not response:
                            response = f"{idx + 1}. [{tag}] Reasons: {sep_reasons.pop(0)}"
                        else:
                            response = f"{response}\n{idx + 1}. [{tag}] Reasons: {sep_reasons.pop(0)}"
                    else:
                        if not response:
                            response = f"{idx + 1}. [Complete]"  # Reasons: {relevance_reason}"
                        else:
                            response = f"{response}\n{idx + 1}. [Complete]"  # Reasons: {relevance_reason}"

            merged_answer_units = "\n".join(self._split_answer(gold_answer))
            # check length of response and merged answer units equal
            if len(response.split("\n")) == len(merged_answer_units.split("\n")):
                # add highlighted text to df
                df.loc[i, f"{aspect}_highlighted"] = response
                df.loc[i, "identified_answer"] = str(merged_answer_units)
            else:
                # print(merged_answer_units)
                print(response)
                print(len(merged_answer_units.split("\n")))
                print(response.split("\n"))
                print("Length of response and merged answer units not equal.")
                continue


        # remove nan rows of highlighted text
        cols = [f"{aspect}_highlighted"]
        df.dropna(subset=cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # filter columns
        final_cols = ["question_text", "identified_answer"] + cols
        df = df[final_cols]
        # shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # print(df.head())
        return df


if __name__ == '__main__':
    data_path = "src/data/annotated_data/complete_data_scores.csv"
    dataset = ErrorDataset(data_path)
    aspect = "incomplete_ans"  # "irrelevance", "incomplete_ans", "reference_example"
    use_all_data = True
    use_reason = True
    add_score = False
    max_correct_answers = None

    # write the question answers to a json file
    examples = []

    complete_df = pd.DataFrame()
    for ans_choice in ["model", "human"]:
        df = dataset.create_tags(
            aspect=aspect,
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
    print(complete_df.shape)
    for i, row in complete_df.iterrows():
        # replace Answer 1 and Answer 2 with answer
        row[f"{aspect}_highlighted"] = row[f"{aspect}_highlighted"].replace("Answer 1", "Answer")
        row[f"{aspect}_highlighted"] = row[f"{aspect}_highlighted"].replace("Answer 2", "Answer")
        row[f"{aspect}_highlighted"] = row[f"{aspect}_highlighted"].replace("answer 1", "answer")
        row[f"{aspect}_highlighted"] = row[f"{aspect}_highlighted"].replace("answer 2", "answer")
        examples.append({
            "instruction": f"{TASK_INSTRUCTION[aspect]}",
            "input": f"Question: {row['question_text']}\nAnswer: {row['identified_answer']}",
            "output": row[f"{aspect}_highlighted"],
        })

    utils.jdump(examples, f"src/data/annotated_data/{aspect}_detection_data_5.jsonl")
