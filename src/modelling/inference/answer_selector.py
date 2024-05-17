import os
from typing import Optional
from tqdm import tqdm
from collections import Counter
import spacy
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import pandas as pd
import datasets
import re
import json

from src.data_creation import utils
from src.modelling.chat_templates import llm_prompts

TASK_INSTRUCTION = "When given a question and answer statements, evaluate whether each given statement provides "  \
                   "sufficient information for answering the question. \n Use the '[Incomplete]' tag to indicate "  \
                   "answer incompleteness, and '[Complete]' tag to indicate completeness, with reasons.\n Please "  \
                   "note that the answer can have single, multiple, or no incomplete statements."


class AnswerSelector:
    def __init__(
            self,
            model_path: str,
            data_path: str,
            dataset: str,
            num_samples: Optional[int] = None,
            seed: int = 42,
            use_vllm: bool = False,
            max_gens: int = 5
    ):
        self.model_path = model_path
        self.data_path = data_path
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed
        self.use_vllm = use_vllm
        self.max_gens = max_gens
        self.nlp = spacy.load("en_core_web_sm")

    def load_test_data(self, num_samples: Optional[int] = None):
        # convert to hf dataset
        list_data_dict = utils.jload(self.data_path)
        # convert to pandas dataframe
        df = pd.DataFrame(list_data_dict)
        print(df.head())

        dataset = datasets.Dataset.from_pandas(df)
        original_columns = dataset.column_names

        def map_dataset(dataset):
            return dataset.map(
                lambda x:
                {
                    "prompt": llm_prompts.create_llama_base_prompt(has_input=True).format_map(x),
                },
                # batched=True,
                remove_columns=original_columns,
            )

        eval_data = map_dataset(dataset)
        return eval_data

    def load_data(self, num_samples: Optional[int] = None):
        # convert to hf dataset
        list_data_dict = utils.jload(self.data_path)
        # convert to pandas dataframe
        df = pd.DataFrame(list_data_dict)

        dataset = datasets.Dataset.from_pandas(df)
        if self.dataset == "held_out":
            dataset = dataset.train_test_split(test_size=0.1, seed=42)["test"]
            # test_data = dataset["test"]
            # split test dataset to validation and test splits
            # eval_test_dataset = test_data.train_test_split(test_size=0.5, seed=42)
            # eval_data = eval_test_dataset["train"]
            # test_data = eval_test_dataset["test"]

        original_columns = dataset.column_names

        def map_dataset(dataset):
            return dataset.map(
                lambda x:
                {
                    "prompt": llm_prompts.create_llama_base_prompt(has_input=True).format_map(x),
                    # "output": x["output"],
                },
                # batched=True,
                remove_columns=original_columns,
            )

        test_data = map_dataset(dataset)
        test_data = test_data.shuffle(seed=42)
        if num_samples:
            # print("Selecting samples...")
            test_data = test_data.select([2, 4])    # range(num_samples))

        return test_data

    def generate(self, generate_kwargs: dict = None):
        results = []
        data = self.load_data(num_samples=self.num_samples)
        # data = self.load_test_data(num_samples=self.num_samples)
        if self.use_vllm:
            # load model
            model = LLM(self.model_path, dtype="half")
            sampling_params = SamplingParams(
                **generate_kwargs,
                n=self.max_gens
            )
            for i in range(self.max_gens):
                for sample in tqdm(data):
                    # print(sample)
                    # print("---"*4)
                    prediction = model.generate([sample["prompt"]], sampling_params)
                    output = prediction[0].outputs[0].text
                    print(output)

        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # replace generation kwargs max_tokens with max_new_tokens
            generate_kwargs["max_new_tokens"] = generate_kwargs.pop("max_tokens")

            model.eval()
            c = 0

            for i, sample in tqdm(enumerate(data)):
                prompt = sample["prompt"]
                preds = []
                # for i in range(self.max_gens):
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                try:
                    outputs = model.generate(inputs, **generate_kwargs, num_return_sequences=self.max_gens)
                    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)   # [len(sample["prompt"]):]
                    # print(predictions)
                    predictions = [prediction[len(prompt):] for prediction in outputs]

                    consistency_score, selected_prediction = self.select_prediction(predictions)
                    # print(consistency_score)
                    # print(selected_prediction)
                    # print("*" * 10)

                    # collate similar responses
                    final_response = self.collate_similar_responses(selected_prediction)
                    results.append({
                        "prompt": sample["prompt"],
                        # "output": sample["output"],
                        "prediction": final_response,
                        "consistency": consistency_score
                        }
                    )
                except Exception as e:
                    print(e)
                    c += 1
                    continue
                # print(results[i]["prompt"])
                # print(results[i]["prediction"])
                # print("*" * 10)
                # c += 1
                # if c == 1:
                #     break

        # save results if save file is not present
        save_path = f"results/llama3_8b_dpo_completeness_feedback_responses_{self.dataset}_seed_{self.seed}_all.jsonl"
        if os.path.exists(save_path):
            print("File already exists")
        else:
            utils.jdump(results, save_path)
        print(f"Results saved to {save_path}")
        print(f"Total errors: {c}")


    def choose_generalization_selfcheck(self, responses):
        from selfcheckgpt.modeling_selfcheck import SelfCheckNgram
        scores = []
        selfcheck_ngram = SelfCheckNgram(n=1)
        for i, response in enumerate(responses):
            others = responses[:i] + responses[i + 1:]
            s = \
            selfcheck_ngram.predict(
                passage=response,
                sentences=[response],
                sampled_passages=others
            )["sent_level"]["max_neg_logprob"][0]
            scores.append(s)

        choice = responses[0]
        best = scores[0]
        for i, r in enumerate(responses):
            if (scores[i] < best):
                best = scores[i]
                choice = r
        return choice

    def choose_generalization(self, premises, responses):
        premise_words = []
        for p in premises:
            words = {}
            for w in re.findall(r'\b\w+\b', p):
                words[w.lower()] = 1
            premise_words.append(words)

        scores = []
        for response in responses:
            words = re.findall(r'\b\w+\b', response)
            n = 0
            s = 0
            for word in words:
                word = word.lower()
                n = n + 1
                for p in premise_words:
                    if word in p:
                        s = s + 1

            if (n > 0):
                scores.append(s / n)
            else:
                scores.append(0)

        choice = None
        best = 0
        for i, r in enumerate(responses):
            if (scores[i] > best):
                best = scores[i]
                choice = r

        if best == 0:
            choice = premises[0]
        # print(scores)
        return choice

    def select_prediction(self, predictions: list):
        # select best prediction
        tag_consistency = 0
        ans_tags = []

        # Extract sentences and tags from predictions
        for prediction in predictions:
            spans = prediction.split("\n")
            tags = ["Complete" if "[Complete]" in span else "Incomplete" for span in spans]
            ans_tags.append(tags)

        # Convert inner lists to tuples to make them hashable
        list_tuples = [tuple(lst) for lst in ans_tags]

        # Dictionary to store the count and first occurrence index of each list
        list_info = {}
        for idx, item in enumerate(list_tuples):
            if item not in list_info:
                list_info[item] = {'count': 1, 'first_occurrence': idx, 'occurrences': [idx]}
            else:
                list_info[item]['count'] += 1
                list_info[item]['occurrences'].append(idx)

        # Get the most common list and its count
        most_common_list, count = max(list_info.items(), key=lambda x: x[1]['count'])

        # Convert back to list format if needed
        most_common_list = list(most_common_list)
        tag_consistency = count['count']/len(predictions)

        # print("Most common list:", most_common_list)
        # print("Tag consistency:", tag_consistency)
        # print("First occurrence index:", count['first_occurrence'])
        # print("All occurrences indices:", count['occurrences'])

        # select predictions with the most common list
        selected_predictions = [predictions[idx] for idx in count['occurrences']]

        choice = self.choose_generalization(selected_predictions, selected_predictions)
        # print(tag_consistency)
        # print(choice)
        return tag_consistency, choice

    def collate_similar_responses(self, response: str):
        filtered_spans = []
        span_index = 0
        text = response.split("\n")
        for i, span in enumerate(text):
            span = re.sub(r'^\d+\.\s*', '', span)
            if "[Incomplete]" in span and span not in filtered_spans:
                filtered_spans.append(span)
            elif "[Complete]" in span:
                filtered_spans.append(span)
            else:
                filtered_spans.append("[Complete]")
        # reform the text with numbered spans
        filtered_text = "\n".join([f"{i+1}. {span}" for i, span in enumerate(filtered_spans)])
        return filtered_text


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="",
                        help="path to the dataset")
    parser.add_argument("--seed", type=int, required=True, default=42,
                        help="random seed")
    parser.add_argument("--model_path", type=str, required=True,
                        default="Llama-2-13b-hf-completeness/llama.sft.deepspeed.tf.completeness.1",
                        help="path to the model")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="number of samples to select")
    parser.add_argument("--use_vllm", action="store_true", default=False,
                        help="whether to use VLLM model")
    parser.add_argument("--max_gens", type=int, default=20,
                        help="number of generations to perform")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    set_seed(args.seed)

    sc = AnswerSelector(
        data_path="src/data/incomplete_ans_detection_data.jsonl",
        # data_path=f"src/data/annotated_data/{args.dataset}_errors_complete_1.jsonl",
        model_path=args.model_path,
        dataset=args.dataset,
        num_samples=args.num_samples,
        seed=args.seed,
        use_vllm=args.use_vllm,
        max_gens=args.max_gens
    )
    generation_kwargs = dict(
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.18,
        max_tokens=1024,
    )
    sc.generate(
        generate_kwargs=generation_kwargs,
    )
