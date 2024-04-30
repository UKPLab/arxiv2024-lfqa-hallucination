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

TASK="no_refine"
DATASETS=(
#"held_out" \
#"asqa" \
"eli5" \
)
SEEDS=(42 0 1)

for SEED in "${SEEDS[@]}"
do
  for DATASET in "${DATASETS[@]}"
  do
      echo "Running ${TASK} for dataset: ${DATASET}."
      python ${BASE_PATH}/src/modelling/self_refine.py \
      --dataset ${DATASET} \
      --task ${TASK} \
      --seed ${SEED}
  done
done
