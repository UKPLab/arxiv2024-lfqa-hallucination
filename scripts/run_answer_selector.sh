#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --job-name=evaluation
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_mem:80gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --nodelist=penelope
#SBATCH --output=feedback_ho.out


# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi


python ${BASE_PATH}/src/modelling/inference/answer_selector.py \
--dataset "held_out" \
--seed 42  \
--model_path "Llama-3-8b-hf-completeness/llama3.8b.sft.dpo.completeness.1" \

# "Llama-2-13b-hf-completeness/llama.sft.deepspeed.tf.completeness.1"