"""
Error detection and refining errors in LLM outputs
"""
import gc
import pandas as pd
import pycountry

import nltk
import re
import ast
import string

# nltk.download('punkt', download_dir="/storage/ukp/work/sachdeva/")
# nltk.download('averaged_perceptron_tagger', download_dir="/storage/ukp/work/sachdeva/")
# nltk.download('maxent_ne_chunker', download_dir="/storage/ukp/work/sachdeva/")


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)

import torch

from typing import Dict, Iterable
from itertools import chain
from langchain.prompts import PromptTemplate


class SelfRefine:
    def __init__(
            self,
            model_path: str,
            device: str = "cuda",
            context_len: int = 2048,
            template: str = "hf"
    ):
        self.model_path = model_path
        self.device = device
        self.context_len = context_len
        self.template = template

    def load_model(self, from_pretrained_kwargs: dict):
        print("MODEL LOADING...")
        revision = from_pretrained_kwargs.get("revision", "main")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                revision=revision,
                trust_remote_code=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=False, revision=revision, trust_remote_code=True
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        except NameError:
            model = AutoModel.from_pretrained(
                self.model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )
        return model, tokenizer

    def _create_prompt(self, system_message, examples, prompt):

        if self.template == "hf":
            prefix = "<|im_start|>"
            suffix = "<|im_end|>\n"
            example_prompt = ""
            if examples:
                num_examples = len(examples)
                for i in range(num_examples):
                    example_user_prompt = prefix + "user\n" + examples[i]["input"] + suffix
                    example_assistant_prompt = prefix + "assistant\n" + examples[i]["output"] + suffix
                    example_prompt += example_user_prompt + example_assistant_prompt

            user_format = prefix + "user\n" + prompt + suffix
            assistant_format = prefix + "assistant\n"
            if system_message:
                sys_format = prefix + "system\n" + system_message + suffix
                input_prompt = sys_format + example_prompt + user_format + assistant_format
            else:
                input_prompt = example_prompt + user_format + assistant_format

        elif self.template == "fastchat":
            from src.prompts.conversation import get_conv_template

            conv = get_conv_template("mistral")
            conv.set_system_message(system_message=system_message)

            for ex in examples:
                conv.append_message(conv.roles[0], ex["input"])
                conv.append_message(conv.roles[1], ex["output"])
            conv.append_message(conv.roles[0], prompt)
            input_prompt = conv.get_prompt()

        elif self.template == "langchain":
            pass
        else:
            print("No matching template found. Select one of hf or langchain")

        messages = [
            {"role": "user", "content": input_prompt},
        ]
        return messages

    def inference(
            self,
            params: Dict,
    ):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
        model, tokenizer = self.load_model(kwargs)
        model.to(self.device)

        # Read parameters
        prompt = params["prompt"]
        system_message = params.get("system_message", "")
        examples = params.get("few_shot_examples", "")
        messages = self._create_prompt(system_message, examples, prompt)

        len_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 0.0))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))

        if self.template == "hf":
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
        elif self.template == "fastchat":
            input_ids = tokenizer(messages[0]["content"], return_tensors="pt").input_ids
        # print(tokenizer.batch_decode(input_ids))
        model_inputs = input_ids.to(self.device)

        generated_ids = model.generate(
            model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        decoded = tokenizer.batch_decode(generated_ids)
        return decoded


def load_data(data_path):

    """
    process annotation data
    :param data_path:
    :return:
    """

    def _correct_text(text):
        # sentences = re.split(r'(?<=[.!?])\s', text)
        # sentences = re.split("[" + string.punctuation + "]+", text)
        # sentences = [item.strip() for item in sentences if item.strip() != ""]
        sentences = re.split(r'(?<=[.!?])', text)
        sentences = [item.strip() for item in sentences if item.strip() != ""]
        corrected_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].capitalize() + sentence[1:]
                corrected_sentences.append(sentence)

        # Join the sentences back into a single text
        corrected_text = ' '.join(corrected_sentences)
        # corrected_text = re.sub(r'(?<=[.!?])\s+(.)', lambda x: x.group(1).capitalize(), text)
        return corrected_text



    country_names = [country.name for country in pycountry.countries]

    # Define a function to capitalize country names
    def _capitalize_country_names(text):
        for country in country_names:
            pattern = re.compile(r'\b' + re.escape(country) + r'\b', re.IGNORECASE)
            text = pattern.sub(lambda match: match.group(0).capitalize(), text)

        return text

    df = pd.read_csv(data_path, sep="\t", index_col=0)
    df[["question_text", "ans1_text", "ans2_text"]] = \
        df[["question_text", "ans1_text", "ans2_text"]].map(_correct_text)
    df[["question_text", "ans1_text", "ans2_text"]] = \
        df[["question_text", "ans1_text", "ans2_text"]].map(_capitalize_country_names)
    # df = df.map(_correct_text)
    # df = df.map(_capitalize_country_names)
    return df


def get_samples(aspect, answer_choice, num_samples=1):
    column_wise_annotation = {}
    columns = [
        "question_text", "ans1_text", "ans2_text", f"{aspect}_span", f"{aspect}_label", "ans1_label", "ans2_label"]
    filtered_df = df[columns]
    filtered_df = filtered_df[
        filtered_df[f"{aspect}_label"].apply(lambda x: len(ast.literal_eval(x)) > 0)]
    # print(filtered_df)
    examples = {
        "question": [],
        "answer": [],
        "error_span": [],
    }
    count = 0
    for i, ex in filtered_df.iterrows():
        question = ex["question_text"]
        if ex["ans1_label"].__contains__(answer_choice) and "answer1" in ast.literal_eval(ex[f"{aspect}_label"]):
            idx = ast.literal_eval(ex[f"{aspect}_label"]).index("answer1")
            span = ast.literal_eval(ex[f"{aspect}_span"])[idx]
            answer = ex["ans1_text"]
            count += 1
        elif ex["ans2_label"].__contains__(answer_choice) and "answer2" in ast.literal_eval(ex[f"{aspect}_label"]):
            idx = ast.literal_eval(ex[f"{aspect}_label"]).index("answer2")
            span = ast.literal_eval(ex[f"{aspect}_span"])[idx]
            answer = ex["ans2_text"]
            count += 1
        else:
            continue
        examples["question"].append(question)
        examples["answer"].append(answer)
        examples["error_span"].append(span)
        if count == num_samples:
            break

    return examples


if __name__ == '__main__':
    import json
    model_path = "mistralai/Mistral-7B-Instruct-v0.1"
    category = "history"
    num_annotator = 3
    prolific_id = "613637a4f7a0e5359082010b"
    base_path = "/storage/ukp/work/sachdeva/research_projects/lfqa-eval/"
    data_path = f"src/data/prolific/results_{category}_tud_{num_annotator}_{prolific_id}/lfqa_pilot_complete.csv"
    df = load_data(data_path=data_path)
    question = df["question_text"][0]
    ans1 = df["ans1_text"][0]
    # print(ans1)
    fact_error_span = df["factuality_span"][0]

    examples = get_samples("factuality", answer_choice="model", num_samples=3)
    # print(examples)
    #
    # print(f"Question: {examples['question'][0]}\nAnswer: {examples['answer'][0]}")
    # print(f"Question: {examples['question'][1]}\nAnswer: {examples['answer'][1]}")
    instruction = "Given an english question and its answer, please give a direct " \
                  "span from the answer that is factually incorrect."

    params = \
        {
            "system_message": "Act as an expert and resourceful fact checker.",
            "few_shot_examples": [
                {
                    "input": f"{instruction}\nQuestion: {examples['question'][0]}\nAnswer: {examples['answer'][0]}\nFactually incorrect span: ",
                    "output": f"{examples['error_span'][0]}"
                },
                {
                    "input": f"{instruction}\nQuestion: {examples['question'][1]}\nAnswer: {examples['answer'][1]}\nFactually incorrect span: ",
                    "output": f"{examples['error_span'][1]}"
                },
                # {
                #     "input": f"{instruction}Question: {examples['question'][2]}\nAnswer: {examples['answer'][2]}",
                #     "output": f"The error span is: '{examples['error_span'][2]}'"
                # },
                # {"input": "what is 2+4?", "output": "6"}
            ],
        "prompt": f"{instruction}\nQuestion: {examples['question'][2]}\nAnswer: {examples['answer'][2]}\nError span: "
    }

    chat = SelfRefine(
        model_path=model_path,
        template="hf",
    )
    x = chat.inference(
        params=params
    )
    print(x)
