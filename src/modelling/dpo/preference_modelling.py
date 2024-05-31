import os
import pandas as pd
import torch

from tqdm import tqdm

from datasets import Dataset
from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, HfArgumentParser, GenerationConfig
from src.modelling.dpo.finetune import smart_tokenizer_and_embedding_resize
from trl import DPOTrainer, ORPOConfig, ORPOTrainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_gpt_prompt(question, num_ans_words):
    gpt4_prompt = f"""
Your task is to answer a question by providing a clear and concise explanation of a complex concept in a way  \
that is accessible for laypeople. The question was posted on the reddit forum Explain Like I'm Five  \
(r/explainlikeimfive). Please keep in mind that the question is not literally meant for 5-year-olds,  \
so you should not answer the question in a way that you are talking to a child. Your answer should be  \
around {num_ans_words} words and should break down the concept into understandable parts, providing relevant examples or \ 
analogies where appropriate. You should also aim to make your explanation easy to follow, using clear and  \
concise language throughout. You answer should maintain accuracy and clarity. When appropriate, you can start  \
with one sentence summarizing the main idea of the answer. \

Question: {question}
"""
    return gpt4_prompt


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    #  A smaller beta parameter (less than 0.1) yields optimal results for chat-oriented datasets.
    #  In contrast, larger beta values (ranging from 0.3 to 0.5) are more effective for datasets
    #  focused on instructional fine-tuning, summarization, and similar tasks.
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    ref_model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the reference model name or path"},
    )
    mode: Optional[str] = field(default="train", metadata={"help": "the mode of the script"})
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    warmup_ratio: Optional[float] = field(default=0.0, metadata={"help": "the warmup ratio"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=64, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=32, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    optimization_method: Optional[str] = field(default="dpo", metadata={"help": "the optimization method"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    seed: Optional[int] = field(default=42, metadata={"help": "seed for the reproducibility"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    run_name: Optional[str] = field(default="dpo_llama2", metadata={"help": "The run name for logging"})
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


class PreferenceModelling:
    def __init__(self):
        parser = HfArgumentParser(ScriptArguments)
        self.args = parser.parse_args_into_dataclasses()[0]

    def _load_data(self, data_path):
        df = pd.read_csv(data_path, delimiter="\t", index_col=0)
        # split to train and test
        dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=self.args.seed)
        return dataset

    def _preprocess(self, data_path):
        def return_prompt_and_responses(samples):
            # print(samples)
            return {
                "prompt": [
                    "Question: " + question + "\n\nAnswer: "
                    for question in samples["question_text"]
                ],
                # "prompt": [
                #     create_llama_prompt(question)  # does not work well
                #     for question in samples["question_text"]
                # ],
                "chosen": samples["preferred_response"],  # rated better
                "rejected": samples["rejected_response"],  # rated worse
            }

        dataset = self._load_data(data_path)  # load the dataset
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # split test dataset to validation and test splits
        eval_test_dataset = test_dataset.train_test_split(test_size=0.5, seed=42)
        eval_dataset = eval_test_dataset["train"]
        test_dataset = eval_test_dataset["test"]

        original_columns = train_dataset.column_names
        train_dataset = train_dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=original_columns
        )
        eval_dataset = eval_dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=original_columns
        )
        test_dataset = test_dataset.map(
            return_prompt_and_responses,
            batched=True,
            remove_columns=original_columns
        )
        return train_dataset, eval_dataset, test_dataset

    def dpo_train(self):

        # 1. Load the train and test dataset
        train_dataset, eval_dataset, test_dataset = self._preprocess(self.args.data_path)

        train_dataset = train_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= self.args.max_length
                      and len(x["prompt"]) + len(x["rejected"]) <= self.args.max_length
        )
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= self.args.max_length
                      and len(x["prompt"]) + len(x["rejected"]) <= self.args.max_length
        )

        if self.args.mode == "train":

            # 2. load a pretrained model
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                # load_in_8bit=True,  # check TODO
                # device_map={'': torch.cuda.current_device()},
            )
            model.config.use_cache = False

            if self.args.ignore_bias_buffers:
                # torch distributed hack
                model._ddp_params_and_buffers_to_ignore = [
                    name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
                ]

            # --- trying using the same model ---
            # model_ref = AutoModelForCausalLM.from_pretrained(
            #     self.args.ref_model_name_or_path,
            #     low_cpu_mem_usage=True,
            #     torch_dtype=torch.float16,
            #     load_in_4bit=True,  # check TODO
            # )
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name_or_path,
                padding="right",
            )  # check
            tokenizer.pad_token = tokenizer.eos_token

            if self.args.optimization_method == "dpo":
                # 3. initialize training arguments:
                training_args = TrainingArguments(
                    per_device_train_batch_size=self.args.per_device_train_batch_size,
                    per_device_eval_batch_size=self.args.per_device_eval_batch_size,
                    num_train_epochs=self.args.num_train_epochs,
                    logging_steps=self.args.logging_steps,
                    save_steps=self.args.save_steps,
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                    gradient_checkpointing=self.args.gradient_checkpointing,
                    learning_rate=self.args.learning_rate,
                    evaluation_strategy="steps",
                    eval_steps=self.args.eval_steps,
                    output_dir=self.args.output_dir,
                    report_to=self.args.report_to,
                    lr_scheduler_type=self.args.lr_scheduler_type,
                    warmup_steps=self.args.warmup_steps,
                    warmup_ratio=self.args.warmup_ratio,
                    optim=self.args.optimizer_type,
                    bf16=True,
                    remove_unused_columns=False,
                    run_name=self.args.run_name,
                    seed=self.args.seed,
                )

                peft_config = LoraConfig(
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=[
                        "q_proj",
                        "v_proj",
                        "k_proj",
                        "out_proj",
                        "fc_in",
                        "fc_out",
                        "wte",
                    ],
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                # 4. initialize the DPO trainer
                trainer = DPOTrainer(
                    model,
                    ref_model=None,
                    args=training_args,
                    beta=self.args.beta,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                    # peft_config=peft_config,
                    max_length=self.args.max_length,
                    max_prompt_length=self.args.max_prompt_length,
                )
            elif self.args.optimization_method == "orpo":
                orpo_args = ORPOConfig(
                    learning_rate=self.args.learning_rate,
                    beta=self.args.beta,
                    lr_scheduler_type=self.args.lr_scheduler_type,
                    max_length=self.args.max_length,
                    max_prompt_length=self.args.max_prompt_length,
                    per_device_train_batch_size=self.args.per_device_train_batch_size,
                    per_device_eval_batch_size=self.args.per_device_eval_batch_size,
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                    optim="paged_adamw_8bit",
                    num_train_epochs=self.args.num_train_epochs,
                    evaluation_strategy="steps",
                    eval_steps=self.args.eval_steps,
                    logging_steps=self.args.logging_steps,
                    warmup_steps=self.args.warmup_steps,
                    report_to=self.args.report_to,
                    output_dir=self.args.output_dir,
                )

                trainer = ORPOTrainer(
                    model=model,
                    args=orpo_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=tokenizer,
                )
            else:
                raise ValueError("Invalid optimization method")

            # 6. train
            trainer.train()
            trainer.save_model(self.args.output_dir)

            # 7. save
            output_dir = os.path.join(self.args.output_dir, "final_checkpoint")
            trainer.model.save_pretrained(output_dir)

        elif self.args.mode == "test":
            print("Testing the model")
            # print(eval_dataset[0])
            model = AutoModelForCausalLM.from_pretrained(
                self.args.output_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map="cuda",
                # load_in_4bit=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name_or_path,
            )  # check
            # tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                    tokenizer=tokenizer,
                    model=model,
                )
            model.eval()
            generation_config = GenerationConfig(
                do_sample=True,
                top_k=1,
                temperature=0.1,
                max_new_tokens=512,
                # pad_token_id=tokenizer.eos_token_id,
            )
            with torch.no_grad():
                results = []
                for sample in tqdm(test_dataset):
                    model_input = tokenizer(
                        sample["prompt"],
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.args.max_length,
                    ).to(model.device)
                    prediction = model.generate(
                        **model_input,
                        generation_config=generation_config,
                    )
                    answer = tokenizer.decode(prediction[0], skip_special_tokens=True)
                    processed_answer = answer[len(sample["prompt"]):]
                    # print("Processed answer: ", processed_answer)
                    results.append(
                        {
                            "prompt": sample["prompt"],
                            "prediction": processed_answer,
                        }
                    )
                # break
            # save the results in jsonl file
            with open(f"results_{self.args.output_dir}.jsonl", "w") as f:
                for result in results:
                    f.write(f"{result}\n")


if __name__ == '__main__':
    pm = PreferenceModelling()
    pm.dpo_train()
