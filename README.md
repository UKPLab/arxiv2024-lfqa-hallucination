## Hallucinations in Long-form Question Answering 

This repository contains the code for 

* Creating an expert-annotated hallucination dataset for long-form question answering
* Methodologies to detect and mitigate hallucinations in long-form question answers

---

### Dataset
The dataset is available in the `src/data` folder. The dataset is available in 3 formats:

1. `complete_data.csv`: Contains the expert annotated data along with the hallucination detection labels, reasons and scores.
2. `preference_data.csv`: Contains the expert annotators' preferences for human and model answers. This dataset is used
for preference optimization.
3. `incomplete_ans_detection_data.csv`: Contains the expert annotated data for incomplete answers. This dataset is used for 
training the incomplete answer detection model. This is used as our error feedback model in the feedback-assisted refinement approach.


### Incomplete Answer Detection Model

1. **Training**: The model can be trained using the following command:

```bash
torchrun --nproc_per_node=2 \
  --master_port=2568 ${BASE_PATH}/src/modelling/dpo/finetune.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path src/data/annotated_data/incomplete_ans_detection_data.jsonl \
  --bf16 True \
  --output_dir llama.sft.deepspeed.tf.completeness.1 \
  --num_train_epochs 3  \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "steps" \
  --eval_steps 50 \
  --save_strategy "steps" \
  --save_steps 300 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --report_to "wandb" \
  --run_name "llama_completeness_sft" \
  --fsdp "full_shard auto_wrap"
```

2. **Inference**: We use consistency sampling to select the best answer from the model predictions. The model can be run using the following command:

```bash
python ${BASE_PATH}/src/modelling/answer_selector.py \
--dataset "held_out" \
--seed 42  \
--model_path "Llama-2-13b-hf-completeness/llama.sft.deepspeed.tf.completeness.1" 
```


### Training the Preference Optimization Model
The model can be trained using the following command:

```bash 
MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="llama2_7B_dpo"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  --master_port=2568 ${BASE_PATH}/src/modelling/llama3_dpo.py \
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
```


### Error-informed Refinement
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

### Citation
If you use this code or the dataset, please cite the following paper:

```
COMING SOON
```