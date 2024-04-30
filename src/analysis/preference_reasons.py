import re
import pandas as pd
from typing import List
from collections import Counter


class AnsPreference:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _load_data(self):
        df = pd.read_csv(self.data_path, sep="\t", index_col=0)
        return df

    def analyze_preference(self):
        df = self._load_data()
        column = "ans_preference_reason"
        # drop rows with no reference example
        df = df[df[column].notna()]
        reasons = df[column].values.tolist()

        paragraph = " ".join(reasons)
        words = re.findall(r'\b\w+\b', paragraph.lower())
        # remove common words
        common_words = ["answer", "and", "to", "of", "a", "in", "that", "is", "for", "it", "on", "with", "as", "was",
                        "the", "1", "2", "question", "i", "answers", "more", "not", "this", "be", "are", "have", "or",
                        "but", "if", "can", "from", "at", "they", "than", "all", "so", "what", "there", "which", "does",
                        "both", "better", "also", "its", "how", "because", "by", "why", "about", "while", "an", "s",
                        "however", "provides", "much", "very", "such", "has", "think", "like", "t", "would", "some",
                        "between"]
        words = [word for word in words if word not in common_words]
        counts = Counter(words)
        print(counts.most_common(100))
        # print(all_reasons)


if __name__ == "__main__":
    data_path = "src/data/annotated_data/complete_data_scores.csv"
    ans_preference = AnsPreference(data_path)
    ans_preference.analyze_preference()
