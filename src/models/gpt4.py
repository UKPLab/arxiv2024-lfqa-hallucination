import jsonlines
import os
from tqdm import tqdm
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_response(prompt, max_tokens, model_name):
    """
    :param prompt:
    :param max_tokens:
    :param model_name:
    :return:

    Sample output
    generated_ans =
    {
          "id": "chatcmpl-7dKbsNzkrAjNumSBMc4IhwCZAEMlo",
          "object": "chat.completion",
          "created": 1689608372,
          "model": "gpt-4-0613",
          "choices": [
            {
              "index": 0,
              "message": {
                "role": "assistant",
                "content": "The phenomena of electric and magnetic fields being perpendicular to each other is a
                fundamental part of how electromagnetic waves work. \n\nLet's consider an analogy to understand
                this better. ...},
              "finish_reason": "stop"
            }
          ],
          "usage": {
            "prompt_tokens": 222,
            "completion_tokens": 247,
            "total_tokens": 469
          }
    }

    """
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return response


def zero_shot_gen(data_path, model_name, category):
    """
    generate zero shot answer from openai models
    :param data_path:
    :param model_name:
    :param category:
    :return:
    """
    c = 0
    with jsonlines.open(f"{data_path}QA_By_Human_Answers/{category}.jsonl", "r") as reader:
        for sample in tqdm(reader):
            max_tokens = int(sample["human_ans_white_space_len"]*1.5)
            c += 1
            generated_ans = get_response(sample["prompt"], max_tokens, model_name)

            sample["zero_shot_ans"] = generated_ans["choices"][0]["message"]["content"]
            sample["zero_shot_ans_white_space_len"] = len(sample["zero_shot_ans"].split())
            # print(sample)
            # print(len(sample["zero_shot_ans"].split()))
            with jsonlines.open(f"{data_path}QA_By_Human_Model_Answers/{category}.jsonl", "a") as writer:
                writer.write(sample)
            # break


def question_rating_gen():
    prompt_template = \
        f"Your task is to evaluate questions "
    pass


if __name__ == '__main__':
    filepath = "/home/rachneet/projects/lfqa-eval/src/data/scraped_eli5/"
    # filepath = "/storage/ukp/work/sachdeva/research_projects/lfqa-eval/src/data/scraped_eli5/"
    # old gpt-4-0314
    zero_shot_gen(filepath, model_name="gpt-4-0314", category="Mathematics")
