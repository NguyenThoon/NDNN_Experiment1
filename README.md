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
  training. It also includes diagnostic direct PINN loss columns.
- `shock_curve.csv`: learned shock location `n(t)` on a uniform time grid.
- `solution_grid.npz`: saved arrays `x`, `t`, `u`, and `shock_x` for downstream
  plotting or analysis.
- `reference_solution_grid.npz`: analytic/reference Experiment 1 solution on the
  same plotting grid.
- `direct_pinn_grid.npz`: direct single-network PINN baseline on the same grid.
- `plot_loss.png`: paper-style semilog plot of the total NDNN loss with circular
  markers.
- `plot_solution_ndnn.png`: heat map of the learned piecewise solution with the
  learned shock overlaid.
- `plot_solution_reference.png`: heat map of the analytic/reference solution.
- `plot_solution_direct_pinn.png`: heat map of the direct PINN baseline.
- `plot_exp1_paper_comparison.png`: three-panel comparison in the style of
  Experiment 1: NDNN solution, solution of reference, and direct PINN solution.
- `plot_shock_curve.png`: learned shock curve `x=n(t)`.

## Notes

This is a compact proof-of-concept implementation, not a tuned reproduction of
every numerical detail in the paper. Qualitatively, the intended behavior is a
left rarefaction, a sharp tracked shock initially near `x=0`, and nonstationary
shock motion after interaction with the rarefaction.

## Consistency With The Paper

The code follows the one-shock formulation in Section 2.1: two space-time
networks solve the conservation law in the mapped subdomains, and one
time-dependent network tracks the discontinuity line through the
Rankine-Hugoniot residual.

Experiment 1 in the paper uses `Omega x [0,T] = (-4,1) x [0,3/4]`,
`f(u)=4u(2-u)`, two hidden layers with 20 neurons, 2500 learning nodes, and
weights `lambda=mu=1/2`, matching the defaults here. The paper also notes that
its numerical networks were designed to satisfy initial and boundary conditions;
this proof-of-concept instead uses the initial-condition penalty requested in
the script specification and does not add a separate boundary-condition loss.
The paper mentions regularizing the rarefaction-forming discontinuity; this
script uses the piecewise initial condition directly.

To mirror Experiment 1's paper figures, the script also trains a direct PINN
baseline: one smooth `(x,t) -> u` network over the whole space-time domain with a
PDE residual and initial-condition penalty. This is expected to smear or distort
the shock compared with the NDNN solution.
