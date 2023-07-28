import os
import pandas as pd
import re
import json
import praw
from praw.models import MoreComments
from datetime import datetime
from collections import defaultdict

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")


reddit = praw.Reddit(
    user_agent="Comment Extraction (by /u/INSERTUSERNAME)",
    client_id=client_id,
    client_secret=client_secret
)


class RedditPost:
    def __init__(self, id, title, selftext, subreddit,
                   full_link, score, link_flair_text):
        self.id = id
        self.title = title
        self.selftext = selftext
        self.subreddit = subreddit
        self.full_link = full_link
        self.score = score
        self.link_flair_text = link_flair_text


def extract_subreddits(subreddit_name, time_after, time_before):
    """
    Extract subreddit posts from reddit
    :param subreddit_name:
    :param time_after:
    :param time_before:
    :return:
    """
    subreddit = reddit.subreddit(subreddit_name)
    question_list = []
    objects = []
    after = int(time_after.timestamp())
    before = int(time_before.timestamp())
    for submission in subreddit.search("Planetary Science", sort="hot", limit=None):
        date = submission.created_utc
        if after < date < before:
            objects.append(submission)

    dict_keys = ['q_id', 'title', 'selftext', 'subreddit',
                 'url', 'score', 'category']
    object_keys = ['id', 'title', 'selftext', 'subreddit',
                   'full_link', 'score', 'link_flair_text']

    for object in objects:

        question = defaultdict(dict)
        for dict_key, object_key in zip(dict_keys[1:], object_keys[1:]):
            q_id = object.id
            try:
                if dict_key == "title":
                    question_text = re.sub(
                        "\{?\(?\[?(E|e)(L|l).{1}5\]?\)?\}?\s?:?-?\s?\.?;?,?-?@?\/?",
                        "",
                        str(getattr(object, object_key, None))
                    )
                    question[q_id][dict_key] = question_text.strip()
                    # print(question[q_id][dict_key])
                else:
                    question[q_id][dict_key] = str(getattr(object, object_key, None)).strip()
            except:
                question[q_id][dict_key] = ""

        question[q_id]['date'] = str(datetime.fromtimestamp(object.created_utc))

        # questions that are removed by the moderators should not be added.
        if question[q_id]['selftext'] == "[removed]" or question[q_id]['selftext'] == "[deleted]":
            continue
        question_list.append(question)

        print(f"Last before {str(datetime.fromtimestamp(object.created_utc))}")

        # Write the questions into a file
        now = str(datetime.now()).split(" ")[0]
        # with open(f"Scraped_ELI5/ELI5_Q_2022-11-31_2023-3-31_scraped_{now}_Physics.jsonl", 'a+') as f:
        with open(f"Scraped_ELI5/ELI5_Q_2023-3-31_2023-6-30_scraped_{now}_Sc.jsonl", 'a+') as f:
            for question in question_list:
                output = json.dumps(question) + "\n"
                f.write(output)

        before = object.created_utc - 1

    print("Scraping is done.")


if __name__ == '__main__':
    # after = datetime(2022, 11, 3)
    # before = datetime(2023, 3, 31)
    after = datetime(2023, 3, 31)
    before = datetime(2023, 6, 30)
    # print(after)
    extract_subreddits(subreddit_name="explainlikeimfive", time_after=after, time_before=before)
    # print(posts.head())
    # print(posts.columns)
