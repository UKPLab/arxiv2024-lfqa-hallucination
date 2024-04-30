import pandas as pd
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Trainer,
)
from dataclasses import dataclass, field
import datasets
from datasets import concatenate_datasets
import copy
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence, Union

import logging

from src.data_creation import utils


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_path: Optional[str] = field(default=None)
    use_special_token: bool = field(
        default=False,
        metadata={
            "help": "Use special command during training."
        },
    )
    not_consider_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Consider special tokens during loss calculations."
        },
    )
    use_context_markups: bool = field(
        default=False,
        metadata={
            "help": "make separated training data."
        },
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={
            "help": "use flash attention."
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    separated: bool = field(
        default=False,
        metadata={
            "help": "make separated training data."
        },
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print("# of new special tokens: {}".format(num_new_tokens))
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        skip_tokens: Sequence[int],
        context_markups: Sequence[int],
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    # special token mask
    if skip_tokens is not None:
        for i, input_id_list in enumerate(input_ids):
            for j, orig_token in enumerate(input_id_list):
                if orig_token in skip_tokens:
                    labels[i][j] = IGNORE_INDEX

    if context_markups is not None:
        for i, (label_id_list, source_len) in enumerate(zip(labels, sources_tokenized["input_ids_lens"])):
            context_start = False
            for j, orig_token in enumerate(label_id_list[source_len:]):
                if context_start is False and orig_token == context_markups[0]:
                    context_start = True
                    start_idx = j + source_len
                    for k, orig_token_2 in enumerate(label_id_list[start_idx:]):
                        if orig_token_2 == context_markups[1]:
                            end_idx = start_idx + k
                    labels[i][start_idx + 1:end_idx] = IGNORE_INDEX
                    context_start = False
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            skip_tokens=None,
            context_markups=None,
            separated=False
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = \
            PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        print(sources)
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")

        data_dict = preprocess(sources, targets, tokenizer, skip_tokens, context_markups=context_markups)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class SupervisedDatasetForDPO(Dataset):
    """Dataset for supervised fine-tuning before preference modelling."""

    def __init__(
            self,
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            skip_tokens=None,
            context_markups=None,
            mode="train",
    ):
        super(SupervisedDatasetForDPO, self).__init__()
        logging.warning("Loading data...")
        df = pd.read_csv(data_path, delimiter="\t", index_col=0)
        # split to train and test
        dataset = datasets.Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
        # split test dataset to validation and test splits
        eval_test_dataset = test_dataset.train_test_split(test_size=0.5, seed=42)
        eval_dataset = eval_test_dataset["train"]
        test_dataset = eval_test_dataset["test"]

        original_columns = train_dataset.column_names

        def map_dataset(dataset, output_key):
            return dataset.map(
                lambda x: {
                    "prompt": [
                        "Question: " + question + "\n\nAnswer: "
                        for question in x["question_text"]
                    ],
                    "output": x[output_key],
                },
                batched=True,
                remove_columns=original_columns,
            )

        if mode == "train":
            processed_data = concatenate_datasets([
                map_dataset(train_dataset, "preferred_response"),
                map_dataset(train_dataset, "rejected_response")
            ])
            processed_data = processed_data.shuffle()
        elif mode == "eval":
            processed_data = concatenate_datasets([
                map_dataset(eval_dataset, "preferred_response"),
                map_dataset(eval_dataset, "rejected_response")
            ])
            # test_dataset = concatenate_datasets([
            #     map_dataset(test_dataset, "preferred_response"),
            #     map_dataset(test_dataset, "rejected_response")
            # ])
            processed_data = processed_data.shuffle()
            # test_dataset = test_dataset.shuffle()

        logging.warning("Formatting inputs...")

        sources = [
            example["prompt"]
            for example in processed_data
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in processed_data]

        logging.warning("Tokenizing inputs... This may take some time...")

        data_dict = preprocess(sources, targets, tokenizer, skip_tokens, context_markups=context_markups)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer,
        data_args,
        skip_tokens=None,
        context_markups=None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        skip_tokens=skip_tokens,
        context_markups=context_markups,
        # mode="train",
    )
    eval_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        skip_tokens=skip_tokens,
        context_markups=context_markups,
        # mode="eval",
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():

    """
    HF SFFT Trainer
     dataset = load_dataset("json", data_files="conversations.json",split="train")

     def formatting_prompts_func(example):
     output_texts = []

     for i in range(len(example['prompt'])):

     text = f"### Input: ```{example['prompt'][i]}```\n ### Output: {example['completion'][i]}"

     output_texts.append(text)

     return output_texts


    trainer = SFTTrainer(
     ...
     train_dataset=dataset,
     formatting_func=formatting_prompts_func,
     ...
    )
    :return: None
    """
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.use_flash_attn:
        from src.modelling import llama_flash_att_monkey_patch as llama_patch
        llama_patch.replace_llama_attn_with_flash_attn()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path if model_args.tokenizer_path is None else model_args.tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if "llama" in model_args.model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    # add special tokens
    if model_args.use_special_token is True:
        if data_args.data_path.__contains__("irrelevance"):
            special_tokens = ["[Irrelevant]", "[/Irrelevant]", "[Relevant]"]
        elif data_args.data_path.__contains__("preference"):
            special_tokens = ["[Evidence1]", "[Evidence2]"]
        else:
            special_tokens = []
        special_token_dict = {
            "additional_special_tokens": special_tokens}
        if tokenizer.pad_token is None:
            special_token_dict["pad_token"] = DEFAULT_PAD_TOKEN
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_token_dict,
            tokenizer=tokenizer,
            model=model,
        )

    else:
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )

    if model_args.use_special_token is True:
        special_tokens = tokenizer.additional_special_tokens
        skip_tokens = []
        context_markups = []
        if model_args.use_context_markups is True:
            for token in ["<paragraph>", "</paragraph>"]:
                context_markups.append(tokenizer.convert_tokens_to_ids(token))
            if model_args.not_consider_special_tokens is True:
                skip_tokens = []
                for token in special_tokens:
                    skip_tokens.append(tokenizer.convert_tokens_to_ids(token))
            else:
                skip_tokens = None
            data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, skip_tokens=skip_tokens,
                                                      context_markups=context_markups)
        else:
            data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
