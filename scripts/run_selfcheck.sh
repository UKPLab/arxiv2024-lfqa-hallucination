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
#SBATCH --output=selfcheck.out


# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

METHOD="qa"

DATASETS=(
"held_out" \
"asqa" \
"eli5" \
"eli5_science" \
"eli5_history" \
)
#MODEL_NAME="llama2_chat_dpo_13_03"

MODEL_NAME="llama2_7b_chat_dpo_13_03"
SEED=42

# split model name with / and get the last element
echo $MODEL_NAME | awk -F'/' '{print $NF}'

MODEL="$(echo $MODEL_NAME | awk -F'/' '{print $NF}')"

for DATASET in "${DATASETS[@]}"
  do
      echo "Running selfcheck for model: ${MODEL_NAME}, dataset: ${DATASET} in normal mode."
      python ${BASE_PATH}/src/evaluation/selfcheck.py \
      --method ${METHOD} \
      --pred_file_path "experiments/results_${MODEL}_${DATASET}_seed_${SEED}.jsonl" \
      --sampled_file_path "experiments/results_${MODEL}_sampled_${DATASET}_seed_${SEED}.jsonl" \
      --output_dir "${MODEL}_selfcheck_${METHOD}_${DATASET}_seed_${SEED}.jsonl"
  done

