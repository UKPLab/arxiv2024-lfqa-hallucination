#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --job-name=eval
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_mem:80gb"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=penelope
#SBATCH --output=inference.out

# load env vars
if [ -f .env ]; then
  export $(echo $(cat .env | sed 's/#.*//g'| xargs) | envsubst)
fi

DATASETS=(
"held_out" \
#"asqa" \
#"eli5" \
#"eli5_science" \
#"eli5_history" \
)
DO_SAMPLE="false"
SEED=42
#MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
#MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_NAME="llama3_8b_instruct_dpo_v1"
MODEL_NAME="llama2_7B_orpo"
# split model name with / and get the last element
echo $MODEL_NAME | awk -F'/' '{print $NF}'
# llama2_chat_dpo_13_03
# mistral_instruct_dpo

# run script
if [ ${DO_SAMPLE} = false ]; then
  for DATASET in "${DATASETS[@]}"
  do
      echo "Running inference for model: ${MODEL_NAME}, dataset: ${DATASET} in normal mode."
      python ${BASE_PATH}/src/modelling/llm_inference.py \
      --dataset ${DATASET} \
      --model_name ${MODEL_NAME} \
      --output_dir "$(echo $MODEL_NAME | awk -F'/' '{print $NF}')_${DATASET}_seed_${SEED}.jsonl"
  done
else
  for DATASET in "${DATASETS[@]}"
  do
      echo "Running inference for model: ${MODEL_NAME}, dataset: ${DATASET} in sampling mode."
      python ${BASE_PATH}/src/modelling/llm_inference.py \
      --dataset ${DATASET} \
      --model_name ${MODEL_NAME} \
      --output_dir "$(echo $MODEL_NAME | awk -F'/' '{print $NF}')_sampled_${DATASET}_seed_${SEED}.jsonl" \
      --do_sample
  done
fi
