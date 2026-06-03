#!/bin/bash -l

module purge
module load extra
module load cuda/11.4
module load gcc/9.2.1
module load miniconda3/py39_4.12.0

cd /rhome/tnguy1340/Github/NDNN_Experiment1

rm -rf .venv
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

# Install CUDA-enabled JAX.
# Your NVIDIA driver supports CUDA 12.8, so use the CUDA 12 wheel.
pip install -U "jax[cuda12]"

# Install project dependencies.
pip install optax numpy matplotlib pandas

python - << 'EOF'
import jax
import jaxlib
print("jax version:", jax.__version__)
print("jaxlib version:", jaxlib.__version__)
print("jax devices:", jax.devices())
print("default backend:", jax.default_backend())
EOF