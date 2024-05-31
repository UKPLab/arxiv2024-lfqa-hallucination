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
#SBATCH --output=self_refine.out


# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

TASK="self_refine"
DATASETS=(
#"baseline" \
"held_out" \
#"asqa" \
#"eli5" \
)
SEEDS=(42 0 1)

# llama3_8b_instruct_dpo_v1
#"llama2_7b_chat_dpo_13_03"
# llama2_13b_dpo_8bit
# meta-llama/Llama-2-7b-chat-hf
# llama2_13b_dpo_full/final_checkpoint
# meta-llama/Meta-Llama-3-8B-Instruct

for SEED in "${SEEDS[@]}"
do
  for DATASET in "${DATASETS[@]}"
  do
      echo "Running ${TASK} for dataset: ${DATASET}."
      python ${BASE_PATH}/src/modelling/self_refine.py \
      --model_path meta-llama/Llama-2-13b-chat-hf \
      --feedback_file_path "llama2_13b_completeness_feedback_responses_${DATASET}_seed_${SEED}.jsonl" \
      --output_dir llama2_13b_baseline_feedback_responses_${DATASET}_seed_${SEED}.json \
      --dataset ${DATASET} \
      --task ${TASK} \
      --seed ${SEED}
  done
done
