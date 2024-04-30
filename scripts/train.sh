#!/bin/sh

#SBATCH --job-name=sft
#SBATCH --output=sft.out
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --partition=yolo
#SBATCH --qos=yolo
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --gpus=2
#SBATCH --constraint="gpu_mem:80gb"
# node name
#SBATCH --nodelist=penelope

# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

# run script
torchrun --nproc_per_node=2 \
  --master_port=2568 ${BASE_PATH}/src/modelling/dpo/finetune.py \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --data_path src/data/annotated_data/incomplete_ans_detection_data.jsonl \
  --bf16 True \
  --output_dir llama2-7b-completeness-sft-test \
  --num_train_epochs 3  \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
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
  --run_name "llama_completeness_sft-test" \
  --fsdp "full_shard auto_wrap"
