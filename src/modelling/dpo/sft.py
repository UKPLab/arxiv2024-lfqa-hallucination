import os
import datasets
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import torch
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    BitsAndBytesConfig,
    LlamaConfig,
    Trainer,
)
import bitsandbytes as bnb
from accelerate import Accelerator, init_empty_weights, infer_auto_device_map
import logging
from trl.trainer import ConstantLengthDataset
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field


from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training
)

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def find_all_linear_names(model):
    import re
    model_modules = str(model.modules)
    pattern = r'\((\w+)\): Linear'
    linear_layer_names = re.findall(pattern, model_modules)

    names = []
    # Print the names of the Linear layers
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))
    return target_modules


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    data_path: Optional[str] = field(default="", metadata={"help": "the data path"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    mode: Optional[str] = field(default="train", metadata={"help": "Train or test the model"})
    use_lora: Optional[bool] = field(default=True, metadata={"help": "whether to train using LoRa"})
    num_bits: Optional[int] = field(default=4, metadata={"help": "the number of bits to use for quantization"})
    lora_rank: Optional[int] = field(default=32, metadata={"help": "the rank for LoRa"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha for LoRa"})


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['prompt']}\n\nAnswer: {example['output']}"
    # print(text)
    return text


def prepare_completeness_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"{example['prompt']}{example['output']}"
    # print(text)
    return text


def create_completeness_data(data_path, tokenizer):
    from src.data_creation import utils
    # data_path = "src/data/annotated_data/incomplete_ans_detection_data_5.jsonl"
    # convert to hf dataset
    list_data_dict = utils.jload(data_path)
    # convert to pandas dataframe
    df = pd.DataFrame(list_data_dict)
    # print(df.head())

    dataset = datasets.Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=42)
    train_data = dataset["train"]
    test_data = dataset["test"]
    # split test dataset to validation and test splits
    eval_test_dataset = test_data.train_test_split(test_size=0.5, seed=42)
    eval_data = eval_test_dataset["train"]
    test_data = eval_test_dataset["test"]

    original_columns = train_data.column_names

    # def map_dataset(dataset):
    #     return dataset.map(
    #         lambda x:
    #         {
    #             "prompt": PROMPT_DICT["prompt_input"].format_map(x),
    #             "output": x["output"],
    #         },
    #         # batched=True,
    #         remove_columns=original_columns,
    #     )

    def map_dataset(dataset):
        return dataset.map(
            lambda x:
            {
                "text": f"{PROMPT_DICT['prompt_input'].format_map(x)}{x['output']}",
            },
            # batched=True,
            remove_columns=original_columns,
        )

    train_data = map_dataset(train_data)
    eval_data = map_dataset(eval_data)
    test_data = map_dataset(test_data)

    train_data = train_data.shuffle(seed=42)  # .select(range(10))
    eval_data = eval_data.shuffle(seed=42)  # .select(range(10))
    test_data = test_data.shuffle(seed=42)  # .select(range(10))

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(eval_data)}")
    # print(f"Sample of the train set: {train_data[0]}")
    # print(f"Sample of the validation set: {eval_data[0]}")

    # chars_per_token = chars_token_ratio(train_data, tokenizer)
    # print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # train_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     train_data,
    #     formatting_func=prepare_completeness_text,
    #     infinite=True,
    #     seq_length=1024,
    #     chars_per_token=chars_per_token,
    # )
    # eval_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     eval_data,
    #     formatting_func=prepare_completeness_text,
    #     infinite=False,
    #     seq_length=1024,
    #     chars_per_token=chars_per_token,
    # )
    # test_dataset = ConstantLengthDataset(
    #     tokenizer,
    #     test_data,
    #     formatting_func=prepare_completeness_text,
    #     infinite=False,
    #     seq_length=1024,
    #     chars_per_token=chars_per_token,
    # )
    return train_data, eval_data, test_data


def create_datasets():
    df = pd.read_csv("src/data/annotated_data/preference_data.csv", delimiter="\t", index_col=0)
    # split to train and test
    dataset = datasets.Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=42)
    train_data = dataset["train"]
    test_data = dataset["test"]
    # split test dataset to validation and test splits
    eval_test_dataset = test_data.train_test_split(test_size=0.5, seed=42)
    eval_data = eval_test_dataset["train"]
    test_data = eval_test_dataset["test"]

    original_columns = train_data.column_names

    def map_dataset(dataset, output_key):
        return dataset.map(
            lambda x: {
                "prompt": x["question_text"],
                "output": x[output_key],
            },
            batched=True,
            remove_columns=original_columns,
        )

    train_data = concatenate_datasets([
        map_dataset(train_data, "preferred_response"),
        map_dataset(train_data, "rejected_response")
    ])

    eval_data = concatenate_datasets([
        map_dataset(eval_data, "preferred_response"),
        map_dataset(eval_data, "rejected_response")
    ])

    train_data = train_data.shuffle(seed=42)        #.select(range(10))
    eval_data = eval_data.shuffle(seed=42)      #.select(range(10))

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(eval_data)}")
    # print(f"Sample of the train set: {train_data[0]}")
    # print(f"Sample of the validation set: {eval_data[0]}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    eval_dataset = ConstantLengthDataset(
        tokenizer,
        eval_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=1024,
        chars_per_token=chars_per_token,
    )
    return train_dataset, eval_dataset


if __name__ == '__main__':
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    if script_args.use_lora:
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=script_args.num_bits == 8,
            load_in_4bit=script_args.num_bits == 4,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        # accelerator = Accelerator()
        base_model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name,
                torch_dtype=compute_dtype,
                quantization_config=bnb_config,
                # device_map={"": Accelerator().process_index},
            )
        # Change the LORA hyperparameters accordingly to fit your use case
        peft_config = LoraConfig(
            r=script_args.lora_rank,
            lora_alpha=script_args.lora_alpha,
            target_modules=find_all_linear_names(base_model),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # base_model = prepare_model_for_kbit_training(base_model)
        # base_model = get_peft_model(base_model, peft_config)
        print_trainable_parameters(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            torch_dtype=torch.float16,
        )
    base_model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_dataset, eval_dataset, test_dataset = create_completeness_data(script_args.data_path, tokenizer)

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config if script_args.use_lora else None,
        dataset_text_field="text",
        packing=script_args.packing,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=1024,
    )
    # print_trainable_parameters(trainer.model)
    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
