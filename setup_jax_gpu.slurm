#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-2:00:00
#SBATCH --mail-type=ALL
#SBATCH --job-name="Setup_JAX_GPU"
#SBATCH -p short_gpu
#SBATCH --gpus-per-task=1

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"

cd /rhome/tnguy1340/Github/NDNN_Experiment1

bash setup_jax_gpu.sh

echo "End: $(date)"