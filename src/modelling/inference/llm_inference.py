from tqdm import tqdm
import ast
import jsonlines

import datasets

from vllm import LLM, SamplingParams
from src.evaluation import tiger_score
from src.modelling.chat_templates.llm_prompts import \
    create_llama_prompt, \
    create_mistral_prompt

import torch
SEED = 42
torch.manual_seed(SEED)


PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def create_llama_prompt_with_input(question: str) -> str:
    instruction = "You are a helpful, respectful and honest assistant. Please answer the given question to " \
                  "the best of your ability. If you are unsure about an answer, truthfully say 'I don't know'."
    x = {"instruction": instruction, "input": question}
    prompt = PROMPT_DICT["prompt_input"].format_map(x)
    return prompt


def predict(question: str, model, sampling_params) -> str:
    """Predicts an answer to a question using the given model.

    Args:
        question (str): The question to be answered.
        model_name (str): The name of the model to be used.
        sampling_params (SamplingParams): The sampling parameters to be used.

    Returns:
        str: The predicted answer to the question.
    """
    if model.get_tokenizer().name_or_path.lower().__contains__("llama"):
        input = create_llama_prompt(question)
    elif model.get_tokenizer().name_or_path.lower().__contains__("mistral"):
        input = create_mistral_prompt(question)
    # input = create_llama_prompt(question)
    preds = model.generate([input], sampling_params)
    return preds[0].outputs[0].text


if __name__ == '__main__':
    # p = create_llama_prompt("hello, how are you doing today?")
    # print(p)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="held_out")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--output_dir", type=str, default="mistral_instruct_held_out.jsonl")
    parser.add_argument("--do_sample", action="store_true")

    args = parser.parse_args()
    # dataset = "held_out"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.1"  #"llama2_chat_dpo"   # "meta-llama/Llama-2-7b-chat-hf"
    # "mistral_instruct_dpo"  # "meta-llama/Llama-2-7b-chat-hf"  # "mistralai/Mistral-7B-Instruct-v0.1"
    # output_dir = f"{args.output_dir}_{args.dataset}.jsonl"
    # random 100 number of samples from hf dataset
    # data = datasets.load_dataset("hf_datasets/eli5", "test_eli5").select(range(100))
    # range 100 gets 0-100 samples but i want 100 random samples

    if args.dataset == "asqa":
        asqa_data = datasets.load_dataset("din0s/asqa")
        data = asqa_data["dev"].shuffle(seed=SEED)#.select(range(100))
    elif args.dataset == "eli5":
        eli5_data = datasets.load_dataset("eli5_category")
        data = eli5_data["test"].shuffle(seed=SEED).select(range(1000))
    elif args.dataset == "eli5_history":
        eli5_data = datasets.load_dataset("Pavithree/eli5")
        data = eli5_data["test"].filter(lambda example: example["subreddit"] == "AskHistorians")
        data = data.shuffle(seed=SEED).select(range(1000))
    elif args.dataset == "eli5_science":
        eli5_data = datasets.load_dataset("Pavithree/eli5")
        data = eli5_data["test"].filter(lambda example: example["subreddit"] == "askscience")
        data = data.shuffle(seed=SEED).select(range(1000))
    elif args.dataset == "held_out":
        file_path = f"results_llama2_base.jsonl"
        data = tiger_score.read_results(file_path)

    # print(data.shape)
    # print(data.column_names)
    # print(data["title"][:5])

    model = LLM(args.model_name)
    # print(model.get_tokenizer().name_or_path)
    sampling_params = SamplingParams(
        temperature=1.0 if args.do_sample else 0.0,   # 0.1
        # top_p=1.0,
        # do_sample=True,
        # top_k=1,
        max_tokens=512,
        skip_special_tokens=True,
        seed=SEED
    )

    results = []

    for sample in tqdm(data):
        preds = []
        if args.dataset.__contains__("eli5"):
            question = sample["title"]
        elif args.dataset == "asqa":
            question = sample["ambiguous_question"]
        elif args.dataset == "held_out":
            sample = ast.literal_eval(sample)
            question = sample["prompt"].split("Question: ")[1].split("\n")[0]

        if args.do_sample:
            print("doing sampling")
            for i in range(10):
                prediction = predict(question, model, sampling_params)
                preds.append(prediction)
            results.append({"prompt": question, "prediction": preds})
        else:
            prediction = predict(question, model, sampling_params)
            results.append({"prompt": question, "prediction": prediction})
        # break

    # save results to jsonl using jsonlines
    with jsonlines.open(f"experiments/results_{args.output_dir}", "w") as writer:
        writer.write_all(results)
