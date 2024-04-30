from src.modelling.chat_templates import llm_prompts
import datasets


def doc_to_target(doc: dict[str, str]) -> str:
    return doc['hallucination']


def compute_metrics(gold_answer: str, prediction: str) -> dict[str, float]:
    is_correct = True

    if ("Yes" in prediction and "No" in prediction) or ("Yes" not in prediction and "No" not in prediction):
        is_correct = False
    elif "Yes" in prediction:
        prediction = "yes"
    elif "No" in prediction:
        prediction = "no"

    is_exact = gold_answer == prediction

    res = {"correctness": 1.0 if is_correct else 0.0}
    if is_correct:
        res["em"] = 1.0 if is_exact else 0.0

    res["acc"] = 1.0 if (is_correct and is_exact) else 0.0

    return res


def process_results(doc: dict[str, str], results: list[str]) -> dict[str, float]:
    # results is e.g., ['Yes']
    gold_list = doc_to_target(doc)
    # gold_list is e.g., 'yes'
    prediction = results[0].strip().split("\n")[0]
    scores = compute_metrics(gold_list, prediction)
    return scores


def main():
    dataset = datasets.load_dataset("pminervini/HaluEval", "qa_samples")
    print(dataset)


if __name__ == '__main__':
    main()
