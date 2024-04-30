import torch
import spacy
import ast
import jsonlines
from tqdm import tqdm
from selfcheckgpt.modeling_selfcheck import (
    SelfCheckMQAG,
    SelfCheckBERTScore,
    SelfCheckLLMPrompt,
)

torch.manual_seed(28)
nlp = spacy.load("en_core_web_sm")


# ignore torch warnings
import warnings
warnings.filterwarnings("ignore")


def read_results(file_path):
    with open(file_path, "r") as f:
        results = f.readlines()
    # results = pd.read_json(file_path, lines=True)
    return results


def selfcheck_scoring(args):
    """
    Self check scoring for LLM generated text
    :param args:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_original = read_results(args.pred_file_path)
    data_sampled = read_results(args.sampled_file_path)

    if args.method == "qa":
        selfcheck_mqag = SelfCheckMQAG(device=device)
        selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
    elif args.method == "prompt":
        llm_model = "meta-llama/Llama-2-13b-chat-hf"
        selfcheck_prompt = SelfCheckLLMPrompt(llm_model, device)

    scores = []
    err_count = 0
    for data_orig, data_sam in tqdm(zip(data_original, data_sampled)):

        passage = ast.literal_eval(data_orig)["prediction"].replace("\n", " ").strip()
        samples = [s.replace("\n", " ").strip() for s in ast.literal_eval(data_sam)["prediction"]]

        sentences = [sent.text.strip() for sent in nlp(passage).sents if len(sent.text.strip()) > 3]

        try:
            if args.method == "qa":
                sent_scores_mqag = selfcheck_mqag.predict(
                    sentences,
                    passage,
                    samples,
                    num_questions_per_sent=5,
                    scoring_method='bayes_with_alpha',
                    beta1=0.95,
                    beta2=0.95,
                )
                sent_scores_bertscore = selfcheck_bertscore.predict(sentences, samples)

                scores.append({
                    "prompt": ast.literal_eval(data_orig)["prompt"],
                    "prediction": ast.literal_eval(data_orig)["prediction"],
                    "mqag": list(sent_scores_mqag),
                    "bertscore": list(sent_scores_bertscore)
                })
            elif args.method == "prompt":
                sent_scores_prompt = selfcheck_prompt.predict(
                    sentences=sentences,
                    sampled_passages=samples,
                    verbose=True,
                )

                scores.append({
                    "prompt": ast.literal_eval(data_orig)["prompt"],
                    "prediction": ast.literal_eval(data_orig)["prediction"],
                    "sent_scores_prompt": list(sent_scores_prompt),
                })
        except Exception as e:
            print(e)
            # count 1 to the number of errors
            err_count += 1
            # continue to next iteration
            continue

    with jsonlines.open(f"experiments/results_{args.output_dir}", "w") as writer:
        writer.write_all(scores)
    print(f"Number of errors: {err_count}")


def compute_selfcheck_score(args):
    selfcheck_results = read_results(f"experiments/results_llama2_chat_selfcheck_{args.method}_{args.dataset}.jsonl")
    dpo_selfcheck_results = read_results(f"experiments/results_llama2_chat_dpo_13_03_selfcheck_{args.method}_{args.dataset}.jsonl")
    # print(selfcheck_results[0])
    mqag_scores = []
    bert_scores = []

    def remove_all_occurrences(lst, num):
        while num in lst:
            lst.remove(num)
        return lst

    count = 0
    bert_count = 0
    average_scores1, average_scores2 = [], []
    avg_ber_scores1, avg_ber_scores2 = [], []
    for result1, result2 in zip(selfcheck_results, dpo_selfcheck_results):
        # print(result)
        # print(result1)
        # print(type(result1))
        if args.method == "qa":
            result1 = ast.literal_eval(result1.replace("NaN", "-1"))
            result2 = ast.literal_eval(result2.replace("NaN", "-1"))
            # print(result1)
            # get average score for each sentence
            mqag1 = remove_all_occurrences(result1["mqag"], -1)   # list of scores for each sentence
            bert1 = remove_all_occurrences(result1["bertscore"], -1)
            # if value >1, change to 1
            bert1 = [1 if score > 1 else score for score in bert1]

            mqag2 = remove_all_occurrences(result2["mqag"], -1)   # list of scores for each sentence
            bert2 = remove_all_occurrences(result2["bertscore"], -1)
            # if value >1, change to 1
            bert2 = [1 if score > 1 else score for score in bert2]

            if sum(mqag1)/len(mqag1) > sum(mqag2)/len(mqag2):
                count += 1
            if sum(bert1)/len(bert1) > sum(bert2)/len(bert2):
                bert_count += 1

            average_scores1.append(sum(mqag1) / len(mqag1))
            average_scores2.append(sum(mqag2) / len(mqag2))

            avg_ber_scores1.append(sum(bert1) / len(bert1))
            avg_ber_scores2.append(sum(bert2) / len(bert2))

        elif args.method == "prompt":
            result1 = ast.literal_eval(result1)
            result2 = ast.literal_eval(result2)
            # get average score for each sentence
            prompt1 = result1["sent_scores_prompt"]
            prompt2 = result2["sent_scores_prompt"]
            if sum(prompt1)/len(prompt1) > sum(prompt2)/len(prompt2):
                count += 1
            # average score for each sentence
            average_scores1.append(sum(prompt1)/len(prompt1))
            average_scores2.append(sum(prompt2)/len(prompt2))


        # get avg of each list and skip nans
        # mqag = [score for score in mqag if not (score != score)]
        # bert = [score for score in bert if not (score != score)
        # print(mqag)
        # print(bert)
        # mqag_avg = sum(mqag) / len(mqag)
        # bert_avg = sum(bert) / len(bert)
        # mqag_scores.append(mqag_avg)
        # bert_scores.append(bert_avg)
        # break

    print(count)
    print(bert_count)
    print(f"Average scores for each sentence: {sum(average_scores1)/len(average_scores1)}")
    print(f"Average scores for each sentence dpo: {sum(average_scores2)/len(average_scores2)}")
    print(f"Average bert scores for each sentence: {sum(avg_ber_scores1)/len(avg_ber_scores1)}")
    print(f"Average bert scores for each sentence dpo: {sum(avg_ber_scores2)/len(avg_ber_scores2)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="qa")
    parser.add_argument("--dataset", type=str, default="held_out")
    parser.add_argument("--pred_file_path", type=str, default="results_llama2_chat_dpo.jsonl")
    parser.add_argument("--sampled_file_path", type=str, default="results_Llama_2_chat_7b_dpo_sampled_held_out.jsonl")
    parser.add_argument("--output_dir", type=str, default="llama2_chat_dpo_selfcheck.jsonl")
    args = parser.parse_args()

    selfcheck_scoring(args)
    # compute_selfcheck_score(args)
