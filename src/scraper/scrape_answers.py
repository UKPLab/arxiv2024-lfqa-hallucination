import os
from datetime import datetime
from collections import defaultdict
import tqdm, json, praw, argparse, pdb, re
from datetime import datetime
import jsonlines
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

skip_list = ["[removed]", "[deleted]", "Your submission has been removed",
             "Welcome to /r/AskHistorians.", "[Read Our Rules]", "This submission has been removed",
             "**Please repost this question", "[deleted by user]"]

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


reddit = praw.Reddit(
    user_agent="Comment Extraction (by /u/INSERTUSERNAME)",
    client_id=client_id,
    client_secret=client_secret
)


def scrape_comments(filepath, category):
    eli5_qs = []
    with open(f"{filepath}Q_By_Categories/{category}.jsonl", 'r') as f:
        eli5_qs.extend([json.loads(x) for x in f.read().strip().split("\n")])

    q_id_list = [f"t3_{y}" for x in eli5_qs for y in x.keys()]
    post_info = reddit.info(fullnames=q_id_list)

    question_comments_list = []  # for the json file
    c = 0
    for post in tqdm.tqdm(post_info):
        # c+=1
        if c<=479:
            continue

        qa_dict = {}
        # print(post)

        title = re.sub("\{?\(?\[?(E|e)(L|l).{1}5\]?\)?\}?\s?:?-?\s?\.?;?,?-?@?\/?", "", post.title)
        if any([x in title for x in skip_list]):
            continue

        selftext = post.selftext
        if any([x in selftext for x in skip_list]):
            continue

        content = {
            "title": title.strip(),
            "selftext": selftext.strip(),
            "subreddit": post.subreddit.display_name,
            "url": post.url,
            "score": post.score,
            "category": post.link_flair_text,
            "date": str(datetime.fromtimestamp(post.created))
        }
        # print(content)
        #
        comments_dict = defaultdict(list)
        post.comments.replace_more(limit=None)

        for comment in post.comments:
            comment_text = comment.body
            if any([x in comment_text for x in skip_list]):
                continue

            tokenized_len = len(tokenizer(comment_text)["input_ids"])
            white_space_len = len(comment_text.split())

            comment_len_text_dict = {"tokenized_len": tokenized_len,
                                     "white_space_len": white_space_len,
                                     "comment": comment_text}

            comments_dict[comment.score].append(comment_len_text_dict)

        if comments_dict == {}:
            continue

        content["comments"] = comments_dict

        qa_dict = {post.id: content}

        output = json.dumps(qa_dict) + "\n"
        now = str(datetime.now()).split(" ")[0]
        with open(f"{filepath}QA_By_Categories/ELI5_QA_{category}_scraped_{now}.jsonl", "a+") as f:
            f.write(output)


if __name__ == '__main__':
    filepath = "/home/rachneet/projects/lfqa-eval/src/data/scraped_eli5/"
    # filepath = "/storage/ukp/work/sachdeva/research_projects/lfqa-eval/src/data/scraped_eli5/"
    scrape_comments(filepath, category="Mathematics")
