from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from sklearn.cluster import KMeans
import jsonlines
import torch
import json
from tqdm import tqdm
import numpy as np
import spacy

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

# assert torch.cuda.is_available() == True

if torch.cuda.is_available():
    PATH = "/storage/ukp/work/sachdeva/research_projects/lfqa-eval/src/data/scraped_eli5/"
else:
    PATH = "/home/rachneet/projects/lfqa-eval/src/data/scraped_eli5/"


def cluster_questions():
    """
    Clustering questions that are stored in eli5 "other" category.
    :return:
    """

    model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    model.cuda()


    data = []
    with jsonlines.open(
            f"{PATH}Q_By_Categories/Other.jsonl",
            "r"
    ) as reader:
        for sample in tqdm(reader):
            data.append(sample)

    q_pool = []
    BATCH_SIZE = 32
    with torch.inference_mode():
        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            minibatch = data[i:i + BATCH_SIZE]
            q_list = [post[next(iter(post))]['title'] for post in minibatch]
            # print(q_list)
            # print(minibatch)
            # break

            q_tokens = tokenizer(q_list, max_length=256, padding='max_length',
                                 truncation=True, return_tensors='pt')
            q_tokens.to('cuda')
            q = model(**q_tokens)
            q_pool.append(q.pooler_output.cpu().numpy())
            # break

        q_pool = np.concatenate(q_pool, axis=0)
        # print(q_pool)

    kmeans = KMeans(n_clusters=20, random_state=0).fit(q_pool)
    labels = kmeans.labels_

    # print(labels)
    # print(data[:3])
    keys = [list(item.keys())[0] for item in data]
    # print(keys)

    with jsonlines.open(f"{PATH}Q_By_Categories/other_eli5_cluster.jsonl", "w") as writer:
        for label, sample, key in zip(labels, data, keys):
            sample[key]["cluster"] = int(label)
            writer.write(sample)


def identify_clusters(cluster_num):
    count = 0
    with jsonlines.open(f"{PATH}Q_By_Categories/other_eli5_{cluster_num}.jsonl", "w") as writer:
        with jsonlines.open(f"{PATH}Q_By_Categories/other_eli5_cluster.jsonl", "r") as reader:
            for sample in tqdm(reader):
                cluster = [sample[next(iter(sample))]['cluster']][0]
                if cluster == cluster_num:
                    # print(sample)
                    print([sample[next(iter(sample))]['title']][0])
                    count += 1
                    writer.write(sample)

    print(count)


def question_entities():
    # Load the SpaCy language model
    nlp = spacy.load("en_core_web_sm")

    count = 1
    data = []
    with jsonlines.open(
            f"{PATH}Q_By_Categories/other_eli5_cluster.jsonl",
            "r"
    ) as reader:
        for sample in tqdm(reader):
            data.append(sample)

    questions = [post[next(iter(post))]['title'] for post in data]
    q_ids = [list(post.keys())[0] for post in data]
    # Analyze each question and extract named entities
    with jsonlines.open(f"{PATH}Q_By_Categories/Other_date.jsonl", "w") as writer:
        for sample, question in zip(data, questions):
            # print("Question:", question)
            doc = nlp(question)

            # Extract named entities and their labels
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Print the entities
            for entity, label in entities:
                if label in ["DATE"]:
                    print(f"Entity: {entity}, Label: {label}")
                    print("-" * 50)
                    count += 1
                    writer.write(sample)
    print(count)


def topic_modelling(mode="test"):

    data = []
    with jsonlines.open(
            f"{PATH}Q_By_Categories/other_eli5_cluster.jsonl",
            "r"
    ) as reader:
        for sample in tqdm(reader):
            data.append(sample)

    q_list = [post[next(iter(post))]['title'] for post in data]
    if mode == "train":
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(q_list)
        # print(topics, probs)
        topic_model.save("topic_modelling", serialization="safetensors")
    model = BERTopic.load("topic_modelling")
    df = model.get_document_info(q_list)
    filtered_df = df[df['Topic'] == 140]
    print(filtered_df)

    # war history 20, 19, 147, 146, 140
    # Law (14)
    # [['court', 0.08052519728403562], ['jury', 0.06006260138691666], ['lawyers', 0.03841275952794479],
    #  ['guilty', 0.036966958827972546], ['lawyer', 0.03373594175853086], ['represent', 0.029198396061933817],
    #  ['cases', 0.027235208140862718], ['attorney', 0.024422818407591883], ['case', 0.021313484854565235],
    #  ['oath', 0.019347613019620017]]


if __name__ == '__main__':
    identify_clusters(cluster_num=2)
    # question_entities()
    # topic_modelling()

# 0: Politics
# 1: Law
# 2: Philosophy, History
# 3, 5, 16: History & Politics
