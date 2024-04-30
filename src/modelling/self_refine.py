"""
Error detection and refining errors in LLM outputs
"""

import pandas as pd
import time
import re
from tqdm import tqdm

# set the torch seed
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
)

from typing import Dict, Iterable, List
from src.data_creation import utils


# def self_refine_prompt():
#     template = """
# Answer the following question: "{question}"
# Your answer is: "{answer}".
# Sentence: "{sentence}" in the answer is not complete because: "{reason}".
# Please improve your answer.
# Your improved answer:
#
# """
#     return template

# try 1
# def self_refine_prompt():
#     template = """
# Answer the following question: "{question}"
# Your answer is: "{answer}".
# The answer is not complete because: "{reason}".
# Please improve your answer.
# Your improved answer:
#
# """
#     return template

# best
def self_refine_prompt():
    template = """
Answer the following question: "{question}"
Your answer is: "{answer}".
The answer is not complete because: 
"{reason}".
Please improve your answer.
Your improved answer:

"""
    return template


# def no_feedback_prompt():
#     template = """
# Given the following question: "{question}"
# The original answer is: "{answer}"
# Please refine the original answer (only if needed) to better answer the question.
# Your refined answer:
#
# """
#     return template

def no_feedback_prompt():
    template = """
Answer the following question: "{question}"
Your answer is: "{answer}".
Please improve your answer.
Your improved answer:

"""
    return template


def generic_feedback_prompt():
    template = """
Answer the following question: "{question}"
Your answer is: "{answer}".
The answer is not complete.
Please improve your answer.
Your improved answer:

"""
    return template


def error_detection_prompt():
    template = """
When given a question and answer statements, evaluate whether each given statement provides sufficient information for answering the question.
Use the '[Incomplete]' tag to indicate answer incompleteness, and '[Complete]' tag to indicate completeness, with reasons.
Please note that the answer can have single, multiple, or no incomplete statements.

#Question#: Why do people often say diet sodas are just as bad or worse than regular sodas?
#Answer#: 1. Diet sodas are often considered just as bad or worse than regular sodas because they contain artificial sweeteners instead of sugar.
2. These sweeteners, like aspartame or sucralose, can have negative effects on the body.
3. For example, they may confuse the body's natural ability to regulate calorie intake by tricking it into thinking it's consuming real sugar.
4. This can lead to overeating and weight gain.
5. Additionally, some studies have suggested that artificial sweeteners might negatively impact gut bacteria, potentially leading to health issues.
6. While diet sodas have fewer calories than regular sodas, they can still contribute to poor overall health due to these artificial ingredients.
7. Therefore, many people believe they are no better than, or even worse than, regular sodas.

#Your evaluation#: 1. [Complete]
2. [Complete]
3. [Complete]
4. [Complete]
5. [Complete]
6. [Incomplete] Reasons: Overall health is a combination of nutrition and exercise, artificial ingredients do not necessarily contribute to poor health.
7. [Complete]

#Question#: {question}
#Answer#: {answer}

#Your evaluation#:

"""
    return template


########################################################################################################################
################################################## END OF PROMPTS ######################################################
########################################################################################################################

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

        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
        self.model, self.tokenizer = self.load_model(kwargs)
        self.model.to(self.device)

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
            # {"role": "assistant", "content": "Sure, the improved answer is:"},
        ]
        return messages

    def inference(
            self,
            params: Dict,
    ):

        # Read parameters
        prompt = params["prompt"]
        system_message = params.get("system_message", "")
        examples = params.get("few_shot_examples", "")

        # start time
        start_time = time.time()
        messages = self._create_prompt(system_message, examples, prompt)
        end_time = time.time()
        # time in seconds
        # print(f"Time taken to create prompt: {end_time-start_time}")

        len_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 0.0))  # -1 means disable
        max_new_tokens = int(params.get("max_new_tokens", 256))

        if self.template == "hf":
            chat_template = open('./src/modelling/chat_templates/llama-2-chat.jinja').read()
            chat_template = chat_template.replace('    ', '').replace('\n', '')
            self.tokenizer.chat_template = chat_template
            input_len = len(self.tokenizer.apply_chat_template(messages, tokenize=False)) - 1
            input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        elif self.template == "fastchat":
            input_ids = self.tokenizer(messages[0]["content"], return_tensors="pt").input_ids
        # print(self.tokenizer.batch_decode(input_ids))
        model_inputs = input_ids.to(self.device)

        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        decoded = [text[input_len:] for text in decoded]
        return decoded


# def process_inputs(task: str, data: List[Dict]):
#     # print(data)
#     results = []
#     for d in data:
#         feedback: Dict[str, Iterable] = {}
#
#         if task == "self_refine":
#             prompt = d["prompt"]
#             start_index = prompt.find("Question:") + len("Question:")
#             end_index = prompt.find("Answer:")
#             question = prompt[start_index:end_index].strip()
#
#             start_index = prompt.find("Answer:") + len("Answer:")
#             end_index = prompt.find("\n### Response")
#             answer = prompt[start_index:end_index].strip()
#
#             feedback["question"] = question
#             feedback["answer"] = answer
#
#             output = d["prediction"]
#             # for sentence in output if it has the [incomplete] tag, save the error sentence and reason
#             error_sentence = ""
#             reason = ""
#             for i, sent in enumerate(output.split("\n")):
#                 if "[Incomplete]" in sent:
#                     error_sentence = i + 1
#                     reason = sent.split("Reasons: ")[1]
#                     break
#
#             feedback["error_sentence"] = error_sentence
#             feedback["reason"] = reason
#         elif task in ["no_refine", "generic_refine"]:
#             feedback["question"] = d["question"]
#             feedback["answer"] = d["answer"]
#         results.append(feedback)
#     # return feedback for all samples
#     return results


def process_inputs(task: str, dataset: str, data: List[Dict]):
    # print(data)
    results = []
    for d in data:
        feedback: Dict[str, Iterable] = {}

        if task == "self_refine" or dataset == "held_out":
            prompt = d["prompt"]
            start_index = prompt.find("Question:") + len("Question:")
            end_index = prompt.find("Answer:")
            question = prompt[start_index:end_index].strip()

            start_index = prompt.find("Answer:") + len("Answer:")
            end_index = prompt.find("\n### Response")
            answer = prompt[start_index:end_index].strip()

            feedback["question"] = question
            # combine the answer sentences into a single string
            pattern = re.compile(r'^\d+\.\s*')
            # Remove numbering from each sentence using regex substitution
            ans_statements = answer.split("\n")
            ans_statements_wo_numbering = [pattern.sub('', statement) for statement in ans_statements]
            combined_answer = " ".join(ans_statements_wo_numbering)
            feedback["answer"] = combined_answer

            output = d["prediction"]
            # for sentence in output if it has the [incomplete] tag, save the error sentence and reason
            error_sentences = []
            reasons = []
            for i, sent in enumerate(output.split("\n")):
                if "[Incomplete]" in sent:
                    error_sentences.append(i + 1)
                    reasons.append(sent.split("Reasons: ")[1])
                    # break

            feedback["error_sentence"] = error_sentences
            feedback["reason"] = reasons
        elif task in ["no_refine", "generic_refine"]:
            feedback["question"] = d["question"]
            feedback["answer"] = d["answer"]
        results.append(feedback)
    # return feedback for all samples
    return results


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, default="no_refine")
    parser.add_argument("--model_path", type=str, required=False, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--base_path", type=str, required=False,
                        default="/storage/ukp/work/sachdeva/research_projects/lfqa-eval/")
    parser.add_argument("--dataset", type=str, required=True, default="held_out")
    parser.add_argument("--seed", type=int, required=True, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # task = "no_refine"  # "error_detection" or "self_refine" or "no_refine" or "generic_refine"
    data = utils.jload(
        # f"{args.base_path}results/llama2_13b_completeness_feedback_responses_{args.dataset}_seed_{args.seed}_all.jsonl"
        f"{args.base_path}src/data/annotated_data/{args.dataset}_errors_complete_1.jsonl"
    )
    feedback_samples = process_inputs(task=args.task, dataset=args.dataset, data=data)
    # print(feedback_samples[0])

    self_refine = SelfRefine(
        model_path=args.model_path,
        template="hf",
    )

    if args.task in ["self_refine", "generic_refine"]:
        system_message = "Your task is to improve your answer based on the given feedback. Please ensure that the new answer is complete and accurate."
    elif args.task == "no_refine":
        system_message = "Your task is to improve your answer. Please ensure that the new answer is accurate."
    elif args.task == "error_detection":
        system_message = "Your task is to act as an answer judge."
    else:
        raise ValueError("Invalid task. Choose either 'error_detection' or 'self_refine'")
    params = {
        "system_message": system_message,
        "examples": [],
        "temperature": 0.1,
        "top_p": 0.9,
        "repetition_penalty": 1.18,
        "max_new_tokens": 1024,
    }
    outputs = []
    for feedback in tqdm(feedback_samples):
        question = feedback["question"]
        answer = feedback["answer"]

        if args.task == "self_refine":
            error_sent = feedback["error_sentence"]
            reason = feedback["reason"]
            # convert list of reasons to a single string with each reason numbered
            reason = "\n".join([f"{i + 1}. {r}" for i, r in enumerate(reason)])
            # print(reason)
            # error_sent_str = answer.split("\n")[error_sent - 1]
            prompt = self_refine_prompt().format(
                question=question, answer=answer, reason=reason
            )
            # print(prompt)
        elif args.task == "no_refine":
            prompt = no_feedback_prompt().format(
                question=question, answer=answer,
            )
        elif args.task == "generic_refine":
            prompt = generic_feedback_prompt().format(
                question=question, answer=answer
            )
        elif args.task == "error_detection":
            prompt = error_detection_prompt().format(
                question=question, answer=answer
            )

        # print(prompt)
        params["prompt"] = prompt

        predictions = self_refine.inference(
            params=params
        )[0]
        # prediction = [pred[len(prompt):] for pred in predictions]
        # print(predictions)
        # print("*" * 100)
        sections = predictions.split('\n\n')
        updated_text = '\n\n'.join(sections[1:])
        # print(predictions)
        # print("*" * 100)
        # print(updated_text)
        # print("#" * 100)

        outputs.append(
            {
                "prompt": prompt,
                "question": question,
                "answer": answer,
                "error_sentence": error_sent if args.task == "self_refine" else None,
                "reason": reason if args.task == "self_refine" else None,
                "prediction": updated_text
            }
        )
        # break
        # save the prediction
        # feedback["prediction"] = prediction
        # break
    # save the outputs
    with open(
            f"{args.base_path}results/llama2_13b_no_feedback_responses_{args.dataset}_seed_{args.seed}_all.json",
            "w") as f:
        utils.jdump(outputs, f)
