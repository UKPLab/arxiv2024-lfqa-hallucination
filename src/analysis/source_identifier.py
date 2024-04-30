import re
import pandas as pd
from ast import literal_eval
from collections import Counter


class SourceIdentifier:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def _load_data(self):
        df = pd.read_csv(self.data_path, sep="\t", index_col=0)
        return df

    def analyze_source(self):
        df = self._load_data()
        column = "incomplete_ans_reason"
        # drop rows with no reference example
        df = df[df[column].notna()]
        reference_example = df[column].values.tolist()
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        # url_pattern = r'https?://\S+'
        # url_pattern = r'(https?://\S+|www\.\S+\.\S+|\S+\.com\S*|\S+\.org\S*|\S+\.net\S*)'
        web_urls = []
        for text in reference_example:
            reason = literal_eval(text)[0]
            urls = re.findall(url_pattern, reason)
            if len(urls) > 0:
                for url in urls:
                    web_urls.append(url)

        print(web_urls)
        print(len(web_urls))

        count = 0
        for url in web_urls:
            if "wikipedia" in url:
                count += 1
        print(count)
        # get most common urls
        most_common_urls = Counter(web_urls).most_common(10)
        print(most_common_urls)


if __name__ == '__main__':
    data_path = "src/data/annotated_data/complete_data_scores.csv"
    source_identifier = SourceIdentifier(data_path)
    source_identifier.analyze_source()
