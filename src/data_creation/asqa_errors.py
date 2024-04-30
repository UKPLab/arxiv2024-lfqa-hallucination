import datasets
import re
from tqdm import tqdm

from src.data_creation import utils

SEED = 42

TASK_INSTRUCTION = "When given a question and answer statements, evaluate whether each given statement provides "  \
                   "sufficient information for answering the question. \n Use the '[Incomplete]' tag to indicate "  \
                   "answer incompleteness, and '[Complete]' tag to indicate completeness, with reasons.\n Please "  \
                   "note that the answer can have single, multiple, or no incomplete statements."


class CreateErrorDataset:
    """
    This function creates a dataset of ASQA errors.
    """
    def __init__(self, dataset, num_samples=1000, save_path="", use_scoring=False):
        self.dataset = dataset
        self.num_samples = num_samples
        self.save_path = save_path
        self.use_scoring = use_scoring

    def load_data(self):

        if self.dataset == "asqa":
            asqa_data = datasets.load_dataset("din0s/asqa")
            data = asqa_data["dev"].shuffle(seed=SEED)  # .select(range(100))
        elif self.dataset == "eli5":
            eli5_data = datasets.load_dataset("eli5_category")
            data = eli5_data["test"].shuffle(seed=SEED)
        else:
            raise ValueError("Invalid dataset")
        return data

    def _split_answer(self, answer):
        """
        Split the answer into sentences
        :param answer:
        :return:
        """
        # sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', answer)
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        # join answer with the sentence number
        sentences = [f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)]
        return sentences

    def create_error_dataset(self):
        data = self.load_data()
        eval_results = []
        count = 0
        for sample in tqdm(data):
            # result = ast.literal_eval(result)

            instruction = """
You are a helpful, respectful and honest assistant. Please answer the given question to the best of your ability.

If you are unsure about an answer, truthfully say "I don't know"
"""
            input_context = ""
            answer = ""
            if self.dataset.__contains__("eli5"):
                input_context = sample["title"]
                answer = sample["answers"]["text"][0]
            elif self.dataset == "asqa":
                input_context = sample["ambiguous_question"]
                answer = sample["annotations"][0]["long_answer"]

            merged_answer_units = "\n".join(self._split_answer(answer))
            if self.use_scoring:
                from tigerscore import TIGERScorer
                scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True)  # on GPU
                results = scorer.score([instruction], [answer], [input_context])
                # print(results)
                # check score
                if results[0]["score"] is None:
                    continue
                if results[0]["score"] < 0:
                    eval_results.append(
                        {
                            "instruction": TASK_INSTRUCTION,
                            "question": input_context,
                            "answer": answer,
                            "input": f"Question: {input_context}\nAnswer: {merged_answer_units}",
                            "error": results[0]["score"],
                        }
                    )
                    count += 1

            else:
                eval_results.append(
                    {
                        "instruction": TASK_INSTRUCTION,
                        "question": input_context,
                        "answer": answer,
                        "input": f"Question: {input_context}\nAnswer: {merged_answer_units}",
                    }
                )
                count += 1
            if count >= self.num_samples:
                break

        # save dataset
        self.save_dataset(eval_results)

    def save_dataset(self, eval_results):
        utils.jdump(eval_results, self.save_path + f"{self.dataset}_errors_complete_1.jsonl")
        print(f"Dataset saved as {self.dataset}_errors.jsonl")


if __name__ == "__main__":
    use_scoring = False
    error_dataset = CreateErrorDataset(
        dataset="eli5",  # "asqa", "eli5"
        num_samples=10000,
        save_path="./src/data/annotated_data/",
        use_scoring=use_scoring
    )
    error_dataset.create_error_dataset()
