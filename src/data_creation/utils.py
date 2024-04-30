import ast

import pandas as pd
import os
import re
import json
from typing import List, Union
import pycountry
import io


# create a function to load csv data from multiple files into a single dataframe
def merge_datasets(data_path: str, file_name: str):

    # List all files in the directory folders
    dir_paths = [os.path.join(data_path, filename)
                 for filename in os.listdir(data_path) if not filename.endswith(".csv")]

    complete_df = pd.DataFrame()
    for dir in dir_paths:
        # check if dir is a directory
        if not os.path.isdir(dir):
            continue
        df = pd.read_csv(
            os.path.join(dir, f"{file_name}.csv"),
            na_values=['NA', 'NaN', '', 'NULL', 'missing', "[]"],
            delimiter="\t",
            index_col=0
        )
        df["category"] = dir.split("/")[-1]
        df.reset_index(drop=True, inplace=True)
        complete_df = pd.concat([df, complete_df], axis=0, ignore_index=True)
    return complete_df


def correct_text(text: Union[List, str]):
    # print(text)
    try:
        text = ast.literal_eval(text)
    except:
        text = text

    def _correction(text):
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls = re.findall(url_pattern, text)

        # Replace URLs with a placeholder to preserve their positions
        placeholder = "###URL###"
        for url in urls:
            text = text.replace(url, placeholder)

        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?])', text)
        corrected_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()

            # Check if the sentence contains a placeholder for URL
            if placeholder in sentence:
                corrected_sentence = sentence.replace(placeholder, urls.pop(0))
                corrected_sentence = corrected_sentence[0].capitalize() + corrected_sentence[1:]
                corrected_sentences.append(corrected_sentence)
                continue  # Skip correction for sentences with URLs

            # remove forward slashes at sentence beginnings
            sentence = re.sub(r'^\/', '', sentence)

            if sentence:
                sentence = sentence[0].capitalize() + sentence[1:]
                corrected_sentences.append(sentence)

        # Join the sentences back into a single text
        corrected_text = ' '.join(corrected_sentences)
        return corrected_text

    if isinstance(text, list):
        corrected_texts = []
        for i, item in enumerate(text):
            corrected_text = _correction(item)
            corrected_texts.append(corrected_text)
        return corrected_texts

    elif isinstance(text, str):
        return _correction(text)


def text_correction(df: pd.DataFrame, columns: List[str]):
    names: list = []
    country_names = [country.name for country in pycountry.countries]
    names.extend(country_names)
    # the spaces in some words are added to avoid capitalizing words that are part of URLs
    names.extend(["usa", "uk", "u.s.a", "u.k", "u.s", "u.k.", "u.s.a.", "u.k.", "europe", "napoleon",
                  "european", "europeans", "google", "facebook", " twitter", "instagram", "youtube",
                  "america", "american", "japanese", "japan", "chinese", "china", "indian", "india", "russian",
                  "nazi", "nazis", "nazi's", "nazi's", "nazis'",
                  "soviets", "soviet", "soviet's", "soviet’s", "soviet's", "soviet’s", "ussr", "ussr's", "ussr’s",
                  "berlin", "tiktok", "california", "i've", "kosovo's", "kosovo", "hiroshima", "nagasaki",
                  "konami", "german", "germany", "germans", "germany's", "german's", "germans'", "germany’s"])
    # capitalize fully words that match the list of words
    word_list_upper = ["sms", "gps", "Gps", "qr", "Tv", "tv", "wi-fi", "wifi", "ww1", "ww2", "i", "3g", "4g", "5g",
                 "pc", "gpu", "cpu", "gpus", "cpus", "ev", "api", "ceo", "ceos", "Wtf", "gcfi", "u. S.", "(ww1)",
                 "(ww2)", "Uk", "u. S", "gfci", "g", "g's", "gif", "(i", "cia", "vr", "irac", "ai", "b. C.", "b. C"]

    def _capitalize_country_names(text: Union[List, str]):
        try:
            text = ast.literal_eval(text)
        except:
            text = text

        # Split the text into sentences
        for name in names:
            pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
            if isinstance(text, list):
                text = [pattern.sub(lambda match: match.group(0).capitalize(), item) for item in text]
            elif isinstance(text, str):
                # print(text)
                text = pattern.sub(lambda match: match.group(0).capitalize(), text)
            else:
                return text
        return text

    def _process_string(text):
        try:
            text = ast.literal_eval(text)
        except:
            text = text
        if isinstance(text, list):
            for i, item in enumerate(text):
                text[i] = ' '.join([word.upper() if word in word_list_upper else word for word in item.split()]).replace("’S", "'s")
            return text
        elif isinstance(text, str):
            return ' '.join([word.upper() if word in word_list_upper else word for word in text.split()]).replace("’S", "'s")
        else:
            return text

    df[columns] = df[columns].map(correct_text)
    df[columns] = df[columns].map(_capitalize_country_names)
    df[columns] = df[columns].map(_process_string)
    return df


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


# function to read a json file
def read_json(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)

    return data


if __name__ == '__main__':
    text = " If molecules vibrate and can come undone if vibrated at the right frequency - how come no one has vibrated a human apart?"
    modified_text = text.replace("'S", "'s")
    print(modified_text)
