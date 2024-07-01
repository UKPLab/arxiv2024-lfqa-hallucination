## Hallucinations in Long-form Question Answering 

This repository contains the code for 

* Creating an expert-annotated hallucination dataset for long-form question answering
* Methodologies to detect and mitigate hallucinations in long-form question answers

---

### Dataset
The dataset is available in the `src/data` folder. The dataset is available in 3 formats:

1. `HaluQuestQA.csv`: Contains the expert annotated data along with the hallucination detection labels, reasons and scores.
2. `preference_data.csv`: Contains the expert annotators' preferences for human and model answers. This dataset is used
for preference optimization.
3. `incomplete_ans_detection_data.csv`: Contains the expert annotated data for incomplete answers. This dataset is used for 
training the incomplete answer detection model. This is used as our error feedback model in the feedback-assisted refinement approach.

#### Structure of HaluQuestQA

![HaluQuestQA](https://github.com/UKPLab/lfqa-hallucination/blob/master/images/haluquestqa_sample.png?raw=true)

---

### Requirements
```bash
pip install -r requirements.txt
```

### Usage
All the scripts are present in the 'src/scripts' folder. The important scripts are described below.

#### <ins>Incomplete Answer Detection Model</ins>

1. **Training**: The model can be trained using the following command:

```bash
MASTER_PORT=4638
MODEL_DIR="Llama-2-13b-hf-completeness" # 7b
run_name="llama.sft.deepspeed.tf.completeness.1" # change this every time you run a new experiment

output_dir="${BASE_PATH}/${MODEL_DIR}/${run_name}"
mkdir -p ${output_dir}

CUDA_VISIBLE_DEVICES=0,1 deepspeed \
    --num_gpus 2 \
    --num_nodes 1 \
    --master_port ${MASTER_PORT} \
    ${BASE_PATH}/src/modelling/dpo/sft.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --data_path "src/data/incomplete_ans_detection_data.jsonl" \
    --bf16 True \
    --use_lora False \
    --output_dir ${output_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 64 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05  \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --deepspeed ${BASE_PATH}/src/modelling/dpo/ds_llama_config.json \
    --run_name ${run_name} \
    --seed 42 
```

2. **Inference**: We use consistency sampling to select the best answer from the model predictions. The model can be run using the following command:

```bash
python ${BASE_PATH}/src/modelling/inference/answer_selector.py \
--dataset "held_out" \
--seed 42  \
--model_path "Llama-2-13b-hf-completeness/llama.sft.deepspeed.tf.completeness.1" 
```

3. **Evaluation**: The model can be evaluated using the following command:

```bash
python ${BASE_PATH}/src/modelling/evaluation/error_model_eval.py \
pred_file_path "results/llama2_13b_completeness_feedback_responses_held_out_seed_42_all.jsonl" \
--mode "exact"  # or "adjacent" or "different"
```

#### <ins>Training the Preference Optimization Model</ins>
The model can be trained using the following command:

```bash 
MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="llama2_7b_dpo"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  --master_port=2568 ${BASE_PATH}/src/modelling/dpo/preference_modelling.py \
  --model_name_or_path $MODEL_NAME \
  --ref_model_name_or_path $MODEL_NAME \
  --data_path ${BASE_PATH}/src/data/preference_data.csv \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 5  \
  --eval_steps 20 \
  --save_steps 20 \
  --learning_rate 5e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --beta 0.1 \
  --mode train \
  --max_prompt_length 512 \
  --max_length 1024 \
  --run_name ${OUTPUT_DIR}
  
  =====LoRA Training=====
CUDA_VISIBLE_DEVICES=0 python ${BASE_PATH}/src/modelling/dpo/preference_modelling.py \
  --model_name_or_path $MODEL_NAME \
  --ref_model_name_or_path $MODEL_NAME \
  --data_path ${BASE_PATH}/src/data/preference_data.csv \
  --output_dir ${output_dir} \
  --lora_r 256 \
  --lora_alpha 128 \
  --num_train_epochs 5  \
  --eval_steps 20 \
  --save_steps 20 \
  --learning_rate 5e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --beta 0.1 \
  --mode train \
  --max_prompt_length 512 \
  --max_length 1024 \
  --run_name ${run_name}
```


#### <ins>Error-informed Refinement</ins>
The error-informed refinement can be run using the following command:

```bash
TASK="self_refine"
DATASETS=(
"baseline" \
"held_out" \
"asqa" \
"eli5" \
)
SEEDS=(42 0 1)

for SEED in "${SEEDS[@]}"
do
  for DATASET in "${DATASETS[@]}"
  do
      echo "Running ${TASK} for dataset: ${DATASET}."
      python ${BASE_PATH}/src/modelling/self_refine.py \
      --dataset ${DATASET} \
      --task ${TASK} \
      --seed ${SEED}
  done
done
```

#### <ins>Hallucination evaluation using TigerScore</ins>

The hallucination detection can be run using the following command:

```bash
DATASETS=(
#"baseline" \
"held_out" \
#"asqa" \
#"eli5" \
#"eli5_science" \
#"eli5_history" \
)

MODEL_NAME="Llama-2-13b-chat-hf"

SEEDS=(42 0 1)

for SEED in "${SEEDS[@]}"
do
  for DATASET in "${DATASETS[@]}"
  do
    echo "Evaluating ${MODEL_NAME} on ${DATASET} with seed ${SEED}"
    python ${BASE_PATH}/src/evaluation/tiger_score.py \
    --model_name $MODEL_NAME \
    --output_dir "$(echo $MODEL_NAME | awk -F'/' '{print $NF}')_${DATASET}_seed_${SEED}.jsonl"  \
    --dataset ${DATASET} \
    --seed ${SEED}
  done
done
```

### Contact

- [Rachneet Sachdeva](https://github.com/Rachneet)
- [UKP lab](https://www.ukp.tu-darmstadt.de/)
- [TU Darmstadt](https://www.tu-darmstadt.de/)

### Disclaimer

> **NOTE**
> This repository contains experimental software and is published for the sole purpose of giving additional background
> details on the respective publication.

### Citation
If you use this code or the dataset, please cite the following paper:

```
COMING SOON
```