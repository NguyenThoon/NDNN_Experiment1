#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-2:00:00
#SBATCH --mail-type=ALL
#SBATCH --job-name="NDNN_Exp1"
#SBATCH -p short_gpu
#SBATCH --gpus-per-task=1

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"

module purge
module load extra
module load cuda/11.4
module load gcc/9.2.1
module load miniconda3/py39_4.12.0

cd /rhome/tnguy1340/Github/NDNN_Experiment1
source .venv/bin/activate

echo "Python path:"
which python

echo "GPU check:"
nvidia-smi

echo "JAX check:"
python - << 'EOF'
import jax
import jaxlib
print("jax version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)
print("jax devices:", jax.devices())
print("default backend:", jax.default_backend())
EOF

RUN_DIR="runs/exp1_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

python ndnn_experiment1.py \
    --epochs 50000 \
    --n-pde 2500 \
    --n-rh 2500 \
    --n-ic 2500 \
    --lr 2e-3 \
    --seed 0 \
    --outdir "$RUN_DIR"

echo "End time: $(date)"