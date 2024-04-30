from tqdm import tqdm
import ast
import jsonlines

import datasets

from vllm import LLM, SamplingParams
from src.evaluation import tiger_score


def create_llama_prompt(question: str) -> str:
    """Creates a prompt for the model to generate an answer to a question.

    Args:
        question (str): The question to be answered.

    Returns:
        str: The prompt to be sent to the GPT3 API.
    """
    return f"""
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Please answer the given question to the best of your ability.

If you are unsure about an answer, truthfully say "I don't know"
<</SYS>>

{question} [/INST]
""".strip()


def create_mistral_prompt(question: str) -> str:
    """Creates a prompt for the model to generate an answer to a question.

    Args:
        question (str): The question to be answered.

    Returns:
        str: The prompt to be sent to the GPT3 API.
    """
    return f"""
[INST] You are a helpful, respectful and honest assistant. Please answer the given question to the best of your ability.

If you are unsure about an answer, truthfully say "I don't know"

{question}
[/INST]
""".strip()


def create_falcon_prompt(question: str) -> str:
    """Creates a prompt for the model to generate an answer to a question.

    Args:
        question (str): The question to be answered.

    Returns:
        str: The prompt to be sent to the GPT3 API.
    """
    return f"""
You are a helpful, respectful and honest assistant. Please answer the given question to the best of your ability.

If you are unsure about an answer, truthfully say "I don't know"

{question}
""".strip()
    # return f"""{question}"""


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
    dataset = "qampari"
    output_dir = f"mistral_instruct_dpo_{dataset}.jsonl"
    model_name = "mistral_instruct_dpo"
    # "llama2_chat_dpo"   # "meta-llama/Llama-2-7b-chat-hf"  # "mistralai/Mistral-7B-Instruct-v0.1"

    if dataset == "asqa":
        asqa_data = datasets.load_dataset("din0s/asqa")
        data = asqa_data["dev"]
    elif dataset == "eli5":
        eli5_data = datasets.load_dataset("eli5")
        data = eli5_data["test_eli5"]
    elif dataset == "eli5_science":
        eli5_data = datasets.load_dataset("eli5")
        data = eli5_data["test_asks"]
    elif dataset == "eli5_history":
        eli5_data = datasets.load_dataset("eli5")
        data = eli5_data["test_askh"]
    elif dataset == "held_out":
        file_path = f"results_llama2_base.jsonl"
        data = tiger_score.read_results(file_path)
    elif dataset == "qampari":
        qampari_data = datasets.load_dataset("iohadrubin/qampari_reranking_bm25", trust_remote_code=True)
        print(qampari_data)
        # qampari_data = datasets.load_dataset("princeton-nlp/ALCE-data", "default", use_legacy_dataset=False)
        # print(qampari_data)
        data = qampari_data["test"]

    # print(data.shape)
    # print(data.column_names)
    # print(data[1]["source"])
    # print(data[1]["meta"])
    # print(data[1]["target"])
    # print(data["title"][:5])

    model = LLM(model_name)
    # print(model.get_tokenizer().name_or_path)
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=1.0,
        # do_sample=True,
        # top_k=1,
        max_tokens=512,
        skip_special_tokens=True,
    )

    results = []
    for sample in tqdm(data):

        if dataset in ["eli5", "eli5_science", "eli5_history"]:
            question = sample["title"]
        elif dataset == "asqa":
            question = sample["ambiguous_question"]
        elif dataset == "held_out":
            sample = ast.literal_eval(sample)
            question = sample["prompt"].split("Question: ")[1].split("\n")[0]

        prediction = predict(question, model, sampling_params)
        # print(prediction)
        results.append({"prompt": question, "prediction": prediction})
        # break

    # save results to jsonl using jsonlines
    with jsonlines.open(f"results_{output_dir}", "w") as writer:
        writer.write_all(results)
