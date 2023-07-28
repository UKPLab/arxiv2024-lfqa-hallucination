'''
This file is written to scrape questions 
from Reddit given a date range and subreddit name.
The answer to the questions are scraped by scrape_answers.py.
'''


from unicodedata import category
import requests, time, json, re, pdb
from datetime import datetime
from collections import defaultdict

# https://github.com/Watchful1/Sketchpad/blob/master/postDownloader.py


def downloadFromUrl(after, before, subreddit):

    print(f"Scraping questions from {subreddit}.")

    filter_string = f"subreddit={subreddit}"
    url = "https://api.pushshift.io/reddit/submission/search?limit=1000&order=desc&{}&since={}&until="
    category_list = []
    after = int(after.timestamp())
    before = int(before.timestamp())
    
    while after < before:
        question_list = []
        new_url = url.format(filter_string, after)+str(before)
        json_text = requests.get(new_url)
        time.sleep(1)
        
        try:
            json_data = json_text.json()
            print(json_data)
        except json.decoder.JSONDecodeError:
            time.sleep(1)
            continue
        
        if 'data' not in json_data:
            break
        objects = json_data['data']
        if len(objects) == 0:
            break
    
        dict_keys = ['q_id', 'title', 'selftext', 'subreddit',
                    'url', 'score', 'category']
        object_keys = ['id', 'title', 'selftext', 'subreddit',
                    'full_link', 'score', 'link_flair_text']
        
        for object in objects:
            question = defaultdict(dict)
            for dict_key, object_key in zip(dict_keys[1:], object_keys[1:]):
                q_id = object["id"]
                try:
                    if dict_key == "title":
                        question_text = re.sub(
                            "\{?\(?\[?(E|e)(L|l).{1}5\]?\)?\}?\s?:?-?\s?\.?;?,?-?@?\/?",
                            "",
                            str(object[object_key])
                        )
                        question[q_id][dict_key] = question_text.strip()
                    else:
                        question[q_id][dict_key] = str(object[object_key]).strip()
                except:
                    question[q_id][dict_key] = ""

            question[q_id]['date'] = str(datetime.fromtimestamp(object['created_utc']))

            # questions that are removed by the moderators should not be added.
            if question[q_id]['selftext'] == "[removed]" or question[q_id]['selftext'] == "[deleted]":
                continue
            question_list.append(question)

        print(f"Last before {str(datetime.fromtimestamp(object['created_utc']))}")

        # Write the questions into a file
        now = str(datetime.now()).split(" ")[0]
        with open(f"Scraped_ELI5/ELI5_Q_2022-11-31_2023-3-31_scraped_{now}.jsonl", 'a+') as f:
            for question in question_list:
                output = json.dumps(question)+"\n"
                f.write(output)

        before = object['created_utc'] - 1

    print("Scraping is done.")
    print(category_list)
    return question_list


if __name__ == '__main__':

    # get all the questions
    after = datetime(2022, 11, 3)
    before = datetime(2023, 3, 31)
    subreddit = "explainlikeimfive"

    question_list = downloadFromUrl(after, before, subreddit)
