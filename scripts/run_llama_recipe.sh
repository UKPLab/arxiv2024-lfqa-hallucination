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
#SBATCH --output=training.out   # change causes issues

# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

output_dir="${BASE_PATH}/Llama-2-13b-hf-completeness/llama2_sft_test"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 \
  --nproc_per_node 4  \
  ${BASE_PATH}/src/modelling/llama_finetune.py \
  --enable_fsdp \
  --use_peft \
  --peft_method lora \
  --model_name meta-llama/Llama-2-13b-hf \
  --pure_bf16 \
  --output_dir ${output_dir} \
  --use_fast_kernels
