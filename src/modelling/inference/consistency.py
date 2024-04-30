import time
import pandas as pd
import datasets
import torch
from tqdm import tqdm
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from sklearn.model_selection import train_test_split

from src.data_creation import utils
from src.modelling.chat_templates import llm_prompts

# set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class SelfConsistency:
    def __init__(
            self,
            data_path: str,
            model_path: str,
    ):
        self.data_path = data_path
        self.model_path = model_path

    def load_data(self, num_samples: Optional[int] = None):
        # convert to hf dataset
        list_data_dict = utils.jload(self.data_path)
        # convert to pandas dataframe
        df = pd.DataFrame(list_data_dict)
        # print(df.head())

        dataset = datasets.Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42)
        train_data = dataset["train"]
        test_data = dataset["test"]
        # split test dataset to validation and test splits
        eval_test_dataset = test_data.train_test_split(test_size=0.5, seed=42)
        eval_data = eval_test_dataset["train"]
        test_data = eval_test_dataset["test"]

        original_columns = train_data.column_names

        def map_dataset(dataset):
            return dataset.map(
                lambda x:
                {
                    "prompt": llm_prompts.create_llama_base_prompt(has_input=True).format_map(x),
                    "output": x["output"],
                },
                # batched=True,
                remove_columns=original_columns,
            )

        test_data = map_dataset(test_data)
        test_data = test_data.shuffle(seed=42)
        if num_samples:
            test_data.select(range(num_samples))

        return test_data

    def generate(
            self, num_samples: Optional[int], generate_kwargs: dict = None, use_vllm: bool = False, max_gens: int = 5):
        data = self.load_data(num_samples=num_samples)
        if use_vllm:
            # load model
            model = LLM(self.model_path, dtype="half")
            sampling_params = SamplingParams(
                **generate_kwargs
            )
            for i in range(max_gens):
                for sample in tqdm(data):
                    # print(sample)
                    # print("---"*4)
                    prediction = model.generate([sample["prompt"]], sampling_params)
                    output = prediction[0].outputs[0].text
                    print(output)

        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # replace generation kwargs max_tokens with max_new_tokens
            generate_kwargs["max_new_tokens"] = generate_kwargs.pop("max_tokens")

            results = []
            model.eval()
            for i in range(max_gens):
                for sample in tqdm(data):
                    inputs = tokenizer.encode(sample["prompt"], return_tensors="pt").to(device)
                    outputs = model.generate(inputs, **generation_kwargs)
                    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(sample["prompt"]):]
                    print(prediction)
                    print("*" * 10)


def main():
    # args = parse_args()
    pass


if __name__ == '__main__':
    model_path = "Llama-2-13b-hf-completeness/llama.sft.deepspeed.tf.completeness.1"
    # model_path = "Llama-2-13b-hf-completeness/llama2.sft.deepspeed.tf.completeness.64_32/final_checkpoint_merged"
    sc = SelfConsistency(
        data_path="src/data/annotated_data/incomplete_ans_detection_data_2.jsonl",
        model_path=model_path,
    )
    generation_kwargs = dict(
        # do_sample=True,
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.2,
        max_tokens=1024,
    )
    sc.generate(
        num_samples=None,
        generate_kwargs=generation_kwargs,
        use_vllm=False,
        max_gens=1
    )
