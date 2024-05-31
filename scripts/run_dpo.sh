#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --job-name=training
#SBATCH --gres=gpu:1
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
MODEL_NAME="meta-llama/Llama-2-13b-chat-hf" # 13b
run_name="llama2.13b.sft.dpo" # change this every time you run a new experiment
output_dir="llama2_13b_dpo_full"   #"Llama-3-8b-hf-completeness/llama3.8b.sft.dpo.completeness.1"

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

## ORPO
#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
#  --master_port=2568 ${BASE_PATH}/src/modelling/dpo/preference_modelling.py \
#  --model_name_or_path $MODEL_NAME \
#  --ref_model_name_or_path $MODEL_NAME \
#  --optimization_method "orpo" \
#  --data_path ${BASE_PATH}/src/data/annotated_data/preference_data_13_03.csv \
#  --output_dir ${output_dir} \
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
#  --run_name ${run_name}
