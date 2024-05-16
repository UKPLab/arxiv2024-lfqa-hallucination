#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --job-name=training
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu_mem:80gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --nodelist=penelope
#SBATCH --output=training.out
#SBATCH --error=training.err

# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

# run script

###### MODELS ######
# mistralai/Mistral-7B-v0.1
# mistralai/Mistral-7B-Instruct-v0.1
# meta-llama/Llama-2-7b-chat-hf
# tiiuae/falcon-7b-instruct
# Llama-2-13b-hf-completeness/llama.sft.deepspeed.tf.completeness.1
# "meta-llama/Meta-Llama-3-8B-Instruct"

####################

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

## ORPO
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
#  --master_port=2568 ${BASE_PATH}/src/modelling/preference_modelling.py \
#  --model_name_or_path $MODEL_NAME \
#  --ref_model_name_or_path $MODEL_NAME \
#  --optimization_method "orpo" \
#  --data_path ${BASE_PATH}/src/data/annotated_data/preference_data_13_03.csv \
#  --output_dir $OUTPUT_DIR \
#  --num_train_epochs 5  \
#  --eval_steps 20 \
#  --save_steps 20 \
#  --learning_rate 8e-6 \
#  --warmup_steps 10 \
#  --lr_scheduler_type "linear" \
#  --beta 0.1 \
#  --mode train \
#  --max_prompt_length 512 \
#  --max_length 1024 \
#  --run_name ${OUTPUT_DIR}
