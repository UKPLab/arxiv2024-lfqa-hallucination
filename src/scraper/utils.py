"""
Utility functions for the scraped data
"""
from tqdm import tqdm
import jsonlines
import ast
import statistics

import os
import os.path
from dotenv import load_dotenv

load_dotenv()

BASE_PATH = os.getenv('BASE_PATH')


def deduplicate(filepath):
    with open(filepath+"Q_By_Categories/History.jsonl") as file:
        data = file.readlines()
    count = 0
    seen_samples = set()
    for sample in tqdm(data):
        example = ast.literal_eval(sample)
        q_id = list(example.keys())[0]

        if q_id not in seen_samples:
            try:
                with jsonlines.open(
                        filepath+"Q_By_Categories/History_dedup.jsonl", "a") as file:
                    file.write(example)
            except:
                count += 1
                continue

            # print(example)
            seen_samples.add(q_id)
    print(count)


def filter_category(filepath, category):
    with jsonlines.open(filepath + "ELI5_scraped_Nov_22_Jun_23_combined.jsonl", "r") as reader:
        for sample in tqdm(reader):
            if list(sample.values())[0]["category"] == category:
                with jsonlines.open(
                        filepath+f"ELI5_Q_2022-11-3_2023-6-30_scraped_2023-07-12_{category}.jsonl", "a") as file:
                    file.write(sample)


def process_qa_pairs(filepath, category):
    dir = f"{filepath}scraped_eli5/QA_By_Human_Answers"
    min_human_ans_len = 50
    max_human_ans_len = 500
    seen_samples = set()
    topk_posts = 150
    data = []

    c = 0
    with jsonlines.open(f"{filepath}scraped_eli5/QA_By_Categories/ELI5_QA_{category}_scraped_2023-07-27.jsonl", "r") as reader:
        for sample in tqdm(reader):
            id = list(sample.keys())[0]
            # print(sample)
            question = f"{sample[id]['title'].strip()} {sample[id]['selftext'].strip()}".strip()
            # get most upvoted answer
            comments = sample[id]["comments"]
            upvotes = list(comments.keys())
            # print(upvotes)
            # upvoted comment is always on top
            upvoted_comment = comments[str(upvotes[0])][0]
            len_comment = upvoted_comment["white_space_len"]
            # print(len_comment)

            prompt_template = \
                f"Your task is to answer a question by providing a clear and concise explanation of a complex " \
                "concept in a way that is accessible for laypeople. The question was posted on the reddit forum " \
                "Explain Like I'm Five (r/explainlikeimfive). Please keep in mind that the question is not literally " \
                "meant for 5-year-olds, so you should not answer the question in a way that you are talking to a " \
                f"child. Your answer should be around {len_comment} words and should break down the concept into " \
                f"understandable parts, providing relevant examples or analogies where appropriate. You should also " \
                f"aim to make your explanation easy to follow, using clear and concise language throughout. Your " \
                f"answer should maintain accuracy and clarity. When appropriate, you can start with one sentence " \
                f"summarizing the main idea of the answer.\n\nQuestion: {question} \n\nAnswer (around {len_comment} " \
                f"words):"

            # print(template)

            content = {
                "q_id": id,
                "upvote": int(max(upvotes)),
                "prompt": prompt_template,
                "human_ans": upvoted_comment["comment"],
                "human_ans_white_space_len": int(len_comment),
            }

            if not os.path.exists(dir):
                os.makedirs(dir)

            if min_human_ans_len <= content["human_ans_white_space_len"] <= max_human_ans_len \
                    and content["q_id"] not in seen_samples:
                seen_samples.add(content["q_id"])
                if len(data) < topk_posts:
                    data.append(content)
                else:
                    min_upvote_post = min(data, key=lambda x: x['upvote'])
                    if content['upvote'] > min_upvote_post['upvote']:
                        data.remove(min_upvote_post)
                        data.append(content)

    for sample in tqdm(data):
        with jsonlines.open(f"{dir}/{category}.jsonl", "a") as writer:
            writer.write(sample)

    vote_stat_list = [ex["upvote"] for ex in data]
    len_stat_list = [ex["human_ans_white_space_len"] for ex in data]

    print(f"""Upvote
            \tMean {statistics.mean(vote_stat_list):.2f}
            \tMedian {statistics.median(vote_stat_list)}
            \tMin {min(vote_stat_list)}
            \tMax {max(vote_stat_list)}
            \tstdev {statistics.stdev(vote_stat_list):.2f}""")

    print(f"""Answer length
            \tMean {statistics.mean(len_stat_list):.2f}
            \tMedian {statistics.median(len_stat_list)}
            \tMin {min(len_stat_list)}
            \tMax {max(len_stat_list)}
            \tstdev {statistics.stdev(len_stat_list):.2f}""")


def select_best_instances(filepath, category):
    topk_posts = 110
    data = []
    with jsonlines.open(f"{filepath}scraped_eli5/QA_By_Human_Model_Answers/{category}.jsonl", "r") as reader:
        for sample in tqdm(reader):
            if len(data) < topk_posts:
                data.append(sample)
            else:
                min_upvote_post = min(data, key=lambda x: x['upvote'])
                if sample['upvote'] > min_upvote_post['upvote']:
                    data.remove(min_upvote_post)
                    data.append(sample)

    for ex in tqdm(data):
        with jsonlines.open(f"{filepath}human_annotations/gpt4/{category.lower()}/{category}_zero_shot.jsonl", "a") as writer:
            writer.write(ex)


if __name__ == '__main__':
    filepath = f"{BASE_PATH}/src/data/"
    # deduplicate(filepath)
    # filter_category(filepath, category="Physics")
    # process_qa_pairs(filepath, category="Mathematics")
    select_best_instances(filepath, category="History")
