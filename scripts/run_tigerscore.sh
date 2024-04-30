#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --job-name=training
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_mem:80gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=penelope
#SBATCH --output=tigerscore.out

# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

DATASETS=(
#"held_out" \
"asqa" \
#"eli5" \
#"eli5_science" \
#"eli5_history" \
)

#MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
#MODEL_NAME="llama2_7b_chat_dpo_13_03"
#MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
#MODEL_NAME="llama2_13b_no_feedback_responses"
#MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_NAME="llama3_8b_instruct_dpo_v1"
#MODEL_NAME="Llama-2-7b-chat-hf"
MODEL_NAME="llama2_7B_instruct_orpo_dpo"
#MODEL_NAME="llama2_7B_orpo"
#MODEL_NAME="mistral_instruct_dpo"
#MODEL_NAME="llama2_chat"
SEEDS=(42)

# mistral_instruct_dpo_13_03_eli5_history_1000
# run script
for SEED in "${SEEDS[@]}"
do
  for DATASET in "${DATASETS[@]}"
  do
    echo "Evaluating ${MODEL_NAME} on ${DATASET} with seed ${SEED}"
    python ${BASE_PATH}/src/evaluation/tiger_score.py \
    --output_dir "$(echo $MODEL_NAME | awk -F'/' '{print $NF}')_${DATASET}_seed_${SEED}.jsonl"  \
    --dataset ${DATASET} \
    --seed ${SEED}  \
    --score
  done
done


# # --output_dir "$(echo $MODEL_NAME | awk -F'/' '{print $NF}')_${DATASET}_seed_${SEED}.jsonl"  \