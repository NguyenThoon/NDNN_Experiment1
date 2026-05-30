# NDNN Experiment 1

Minimal Python/JAX implementation of the one-shock non-diffusive neural network
method from Section 2.1 of "Non-diffusive neural network method for hyperbolic
conservation laws."

The script solves

```text
u_t + (f(u))_x = 0,       f(u) = 4u(2-u)
```

on `(-4, 1) x [0, 3/4]` with the three-state initial data from Experiment 1.
It uses two solution networks, `N_minus` and `N_plus`, separated by one learned
shock curve `n(t)` with `n(0)=0`.

## Local Run

Create an environment, install dependencies, and run a small proof-of-concept
training job:

```bash
pip install -r requirements.txt
python ndnn_experiment1.py --epochs 5000 --n-pde 1000 --n-rh 1000 --n-ic 1000 --lr 2e-3 --outdir runs/local_test
```

For a larger run closer to the intended experiment:

```bash
python ndnn_experiment1.py --epochs 50000 --n-pde 2500 --n-rh 2500 --n-ic 2500 --lr 2e-3 --outdir runs/exp1_full
```

## HPCC Submission

Submit the included Slurm script:

```bash
sbatch submit_exp1.slurm
```

The script creates a clean output directory:

```text
runs/exp1_${SLURM_JOB_ID}
```

The module names in `submit_exp1.slurm` are placeholders for UCR HPCC-style use.
Adjust the partition, CUDA module, Python environment activation, or JAX install
method as needed for the specific cluster environment.

## Output Files

- `loss_history.csv`: logged total loss and individual loss terms during
  training.
- `shock_curve.csv`: learned shock location `n(t)` on a uniform time grid.
- `solution_grid.npz`: saved arrays `x`, `t`, `u`, and `shock_x` for downstream
  plotting or analysis.
- `plot_loss.png`: semilog plot of total, PDE, Rankine-Hugoniot, and initial
  condition losses.
- `plot_solution_ndnn.png`: heat map of the learned piecewise solution with the
  learned shock overlaid.
- `plot_shock_curve.png`: learned shock curve `x=n(t)`.

## Notes

This is a compact proof-of-concept implementation, not a tuned reproduction of
every numerical detail in the paper. Qualitatively, the intended behavior is a
left rarefaction, a sharp tracked shock initially near `x=0`, and nonstationary
shock motion after interaction with the rarefaction.
