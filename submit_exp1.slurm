#!/bin/bash
#SBATCH --job-name=ndnn_exp1
#SBATCH --output=ndnn_exp1_%j.out
#SBATCH --error=ndnn_exp1_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load extra
module load cuda/11.4
module load gcc/9.2.1

RUN_DIR="runs/exp1_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

python ndnn_experiment1.py \
  --epochs 50000 \
  --n-pde 2500 \
  --n-rh 2500 \
  --n-ic 2500 \
  --lr 2e-3 \
  --outdir "${RUN_DIR}"
