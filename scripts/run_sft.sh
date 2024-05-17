#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --job-name=training
#SBATCH --gres=gpu:3
#SBATCH --constraint="gpu_mem:80gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --nodelist=penelope
#SBATCH --output=training.out

# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

# inline wrt other papers
# lr: cosine
# max context length: 1024
# lr: 2e-5
# warmup ratio: 0.1
# epochs: 3
# batch size: 128
# 4 A100s
# inference on single A100 with VLLM for faster speeds

MASTER_PORT=4638
MODEL_DIR="Llama-3-8b-hf-completeness" # 13b
run_name="llama3.8b.sft.completeness.1" # change this every time you run a new experiment

#MODEL_DIR="Mistral-7b-completeness"
#run_name="mistral.7b.sft.completeness"

#MODEL_DIR="Gemma-7b-completeness"
#run_name="gemma.7b.sft.completeness"

output_dir="${BASE_PATH}/${MODEL_DIR}/${run_name}"
mkdir -p ${output_dir}


# slurm system gpus can't connect to each other by default
# set the following environment variables to enable nccl
export NCCL_IB_DISABLE=1;
export NCCL_P2P_DISABLE=1;

export NCCL_DEBUG=INFO;
export NCCL_SOCKET_IFNAME=en,eth,em,bond;
export CXX=g++;

#---------------------------------------------------------------
########################## Deepspeed ##########################
# --------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0,1,2 deepspeed \
    --num_gpus 3 \
    --num_nodes 1 \
    --master_port ${MASTER_PORT} \
    ${BASE_PATH}/src/modelling/dpo/sft.py \
    --model_name meta-llama/Meta-Llama-3-8B \
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
    --seed 42 \


# lora lr 1e-4
# lr 2e-4
# warmup 0.03
# weight decay 0.001

# sft lr 2e-5