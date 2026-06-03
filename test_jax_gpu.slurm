#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-2:00:00
#SBATCH --mail-type=ALL
#SBATCH --job-name="JAX_GPU_Test"
#SBATCH -p short_gpu
#SBATCH --gpus-per-task=1

echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"

module purge
module load extra
module load cuda/11.4
module load gcc/9.2.1

# Try loading Python/Miniconda
module load miniconda3/py39_4.12.0

echo "Loaded modules:"
module list

echo "Python path:"
which python

echo "Pip path:"
which pip

echo "Python version:"
python --version

echo "NVIDIA GPU check:"
nvidia-smi

echo "JAX device check:"
python - << 'EOF'
import jax
import jaxlib
print("jax version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)
print("jax devices:", jax.devices())
print("default backend:", jax.default_backend())
EOF

echo "End time: $(date)"