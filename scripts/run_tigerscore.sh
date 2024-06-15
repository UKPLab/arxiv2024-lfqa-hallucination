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
#"baseline" \
#"held_out" \
#"asqa" \
"eli5" \
#"eli5_science" \
#"eli5_history" \
)

#MODEL_NAME="Llama-2-13b-chat-hf"
MODEL_NAME="llama2_13b_error_feedback_responses"
#MODEL_NAME="llama2_13b_dpo_8bit_256_128"
#MODEL_NAME="llama2_7b_chat_dpo_13_03"
#MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
#MODEL_NAME="llama2_13b_dpo_8bit_512_256_error_feedback_responses"
#MODEL_NAME="llama3_8b_dpo_completeness_error_feedback_responses"
#MODEL_NAME="fllama2_7b_dpo_error_feedback_responses"
#MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct-v1"
#MODEL_NAME="Meta-Llama-3-8B-Instruct-dpo-v1"
#MODEL_NAME="llama3_8b_instruct_dpo_v1"
#MODEL_NAME="Llama-2-7b-chat-hf"
#MODEL_NAME="llama2_7B_instruct_orpo_dpo"
#MODEL_NAME="llama2_7B_orpo"
#MODEL_NAME="mistral_instruct_dpo"
#MODEL_NAME="llama2_chat"
SEEDS=(42 0 1)

# mistral_instruct_dpo_13_03_eli5_history_1000
# run script
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

# --output_dir "$(echo $MODEL_NAME | awk -F'/' '{print $NF}')_${DATASET}_seed_${SEED}.jsonl"  \