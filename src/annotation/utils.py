import os
import json
import random
import uuid
import re
import ast


import pandas as pd

def convert_excel_to_text(file_path, file_name):
    """Converts a tsv file to a text file"""

    df = pd.read_csv(file_path+file_name, sep='\t', encoding='utf-8', header=0)

    # read each row of the dataframe and write it to a text file
    documents = [f"{file_name.replace('.tsv', str('_')+str(i+1)+'.txt')}"
                 for i in range(len(df.index))]
    c = 0

    for index, row in df.iterrows():
        question = f"QUESTION:\n{row['question_text'].capitalize()}".replace('<br />', '\n').replace('\r', '')
        answer = f"ANSWER1:\n{row['answer1'].capitalize()}".replace('<br />', '\n').replace('\r', '')
        answer2 = f"ANSWER2:\n{row['answer2'].capitalize()}".replace('<br />', '\n').replace('\r', '')

        save_path = file_path+file_name.split('_')[0].lower()+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path+documents[index], 'w') as f:
            f.write(question+'\n\n'+answer+'\n\n'+question+'\n\n'+answer2)


def convert_jsonl_to_text(file_path, file_name, version):
    """
    Converts a jsonl file to a text file
    :param file_path:
    :param file_name:
    :return:
    """
    with open(file_path+file_name, 'r') as f:
        data = f.readlines()
    documents = [f"{file_name.replace('.jsonl', str('_')+str(i+1)+'.txt')}"
                 for i in range(len(data))]
    data_config = []
    for i in range(len(data)):
        example = json.loads(data[i])
        question = f"QUESTION:\n{example['question_text'].capitalize()}".replace('<br />', '\n').replace('\r', '')
        random_no = random.randint(1, 2)
        human_answer = f"ANSWER{random_no}:\n{example['human_ans'].capitalize()}".replace('<br />', '\n').replace(
            '\r', '')
        save_path = file_path + file_name.split('_')[1].lower() + "/" + file_name.split('_')[0].lower() + "/"
        # print(save_path)
        if version:
            save_path += version+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unique_id = str(uuid.uuid4())
        if random_no == 1:
            model_answer = f"ANSWER2:\n{example['zero_shot_ans'].capitalize()}".replace('<br />', '\n').replace('\r', '')
            with open(save_path + unique_id + '.txt', 'w') as f:
                f.write(question + '\n\n' + human_answer + '\n\n'+ question +'\n\n'+ model_answer)
            # save metadata
            metadata = {unique_id: {"Answer1": "human_answer", "Answer2": "model_answer"}}
        else:
            model_answer = f"ANSWER1:\n{example['zero_shot_ans'].capitalize()}".replace('<br />', '\n').replace('\r', '')
            with open(save_path + unique_id + '.txt', 'w') as f:
                f.write(question + '\n\n' + model_answer + '\n\n'+ question +'\n\n'+ human_answer)
            # save metadata
            metadata = {unique_id: {"Answer1": "model_answer", "Answer2": "human_answer"}}

        data_config.append(metadata)
        # break
    # print(data_config)
    with open(save_path + 'metadata.json', 'w') as f:
        json.dump(data_config, fp=f, indent=4)


def convert_jsonl(file_path, file_name, version):
    """
    Converts a jsonl file to a text file
    :param file_path:
    :param file_name:
    :return:
    """
    with open(file_path+file_name, 'r') as f:
        data = f.readlines()
    documents = [f"{file_name.replace('.jsonl', str('_')+str(i+1)+'.txt')}"
                 for i in range(len(data))]
    data_config = []
    for i in range(len(data)):
        example = json.loads(data[i])
        prompt = example["prompt"]
        question = prompt.split("\n\n")[1].replace("Question: ", "")
        print(question)
        question = f"QUESTION:\n{question.capitalize()}".replace('<br />', '\n').replace('\r', '')
        random_no = random.randint(1, 2)
        human_answer = f"ANSWER{random_no}:\n{example['human_ans'].capitalize()}".replace('<br />', '\n').replace(
            '\r', '')
        save_path = file_path + file_name.split('_')[1].lower() + "/" + file_name.split('_')[0].lower() + "/"
        # print(save_path)
        if version:
            save_path += version+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        unique_id = str(uuid.uuid4())
        if random_no == 1:
            model_answer = f"ANSWER2:\n{example['zero_shot_ans'].capitalize()}".replace('<br />', '\n').replace('\r', '')
            with open(save_path + f"{i}_" + unique_id + '.txt', 'w') as f:
                f.write(question + '\n\n' + human_answer + '\n\n'+ question +'\n\n'+ model_answer)
            # save metadata
            metadata = {unique_id: {"Answer1": "human_answer", "Answer2": "model_answer"}}
        else:
            model_answer = f"ANSWER1:\n{example['zero_shot_ans'].capitalize()}".replace('<br />', '\n').replace('\r', '')
            with open(save_path + f"{i}_" + unique_id + '.txt', 'w') as f:
                f.write(question + '\n\n' + model_answer + '\n\n'+ question +'\n\n'+ human_answer)
            # save metadata
            metadata = {unique_id: {"Answer1": "model_answer", "Answer2": "human_answer"}}

        data_config.append(metadata)
        # break
    # print(data_config)
    with open(save_path + 'metadata.json', 'w') as f:
        json.dump(data_config, fp=f, indent=4)


if __name__ == '__main__':
    # save_path = './src/data/prolific/results_bio_tud_1/'
    filepath = './src/data/human_annotations/gpt4/history/'
    file_name = 'History_zero_shot.jsonl'
    # # convert_excel_to_text(filepath, file_name)
    convert_jsonl(filepath, file_name, version='v0')
    # data_path = 'src/data/human_annotations/gpt3/biology/zero/biology/v0/'
    # data_path = 'src/data/human_annotations/gpt4/biology/zero/biology/v0/'
    # collate_metadata(data_path, save_path)
    # collate_annotations(save_path)
