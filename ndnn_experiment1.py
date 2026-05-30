"""Experiment 1: one-shock NDNN for a scalar conservation law.

This script implements the Section 2.1 one-shock formulation from
"Non-diffusive neural network method for hyperbolic conservation laws" in a
compact, reproducible JAX/Optax training loop.
"""

import argparse
import csv
import os

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd


A = -4.0
B = 1.0
T_FINAL = 0.75
LAMBDA_PDE = 0.5
MU_RH = 0.5
KAPPA_IC = 1.0


def flux(u):
    return 4.0 * u * (2.0 - u)


def u0(x):
    return jnp.where(x < -2.0, 1.0, jnp.where(x < 0.0, 0.5, 1.5))


def init_mlp_params(key, layer_sizes):
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)
    for subkey, in_dim, out_dim in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        limit = jnp.sqrt(6.0 / (in_dim + out_dim))
        w_key, _ = jax.random.split(subkey)
        w = jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim,))
        params.append((w, b))
    return params


def mlp_apply(params, inputs):
    z = inputs
    for w, b in params[:-1]:
        z = jnp.tanh(z @ w + b)
    w, b = params[-1]
    return z @ w + b


def solution_net(params, x, t):
    inputs = jnp.stack([x, t])
    return mlp_apply(params, inputs).squeeze()


def raw_shock_net(params, t):
    inputs = jnp.array([t])
    return mlp_apply(params, inputs).squeeze()


def shock_position(params, t):
    return t * raw_shock_net(params, t)


def init_params(seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    return {
        "minus": init_mlp_params(keys[0], [2, 20, 20, 1]),
        "plus": init_mlp_params(keys[1], [2, 20, 20, 1]),
        "shock": init_mlp_params(keys[2], [1, 20, 20, 1]),
    }


def make_training_points(seed, n_pde, n_rh, n_ic):
    key = jax.random.PRNGKey(seed)
    key_pde, key_rh, key_ic = jax.random.split(key, 3)

    pde_raw = jax.random.uniform(key_pde, (n_pde, 2))
    pde_xi = pde_raw[:, 0]
    pde_t = T_FINAL * pde_raw[:, 1]

    rh_t = T_FINAL * jax.random.uniform(key_rh, (n_rh,))
    ic_xi = jax.random.uniform(key_ic, (n_ic,))

    return {
        "pde_xi": pde_xi,
        "pde_t": pde_t,
        "rh_t": rh_t,
        "ic_xi": ic_xi,
    }


def t_minus_x(shock_x, xi):
    return (shock_x - A) * xi + A


def t_plus_x(shock_x, xi):
    return (B - shock_x) * xi + shock_x


def pde_residual(net_params, x, t):
    # Physical residual: d_t N(x,t) + d_x f(N(x,t)).
    dt_n = jax.grad(lambda tau: solution_net(net_params, x, tau))(t)
    dx_flux = jax.grad(lambda y: flux(solution_net(net_params, y, t)))(x)
    return dt_n + dx_flux


def losses(params, data):
    pde_xi = data["pde_xi"]
    pde_t = data["pde_t"]
    rh_t = data["rh_t"]
    ic_xi = data["ic_xi"]

    shock_at_pde = jax.vmap(lambda tt: shock_position(params["shock"], tt))(pde_t)
    x_minus = t_minus_x(shock_at_pde, pde_xi)
    x_plus = t_plus_x(shock_at_pde, pde_xi)

    # PDE losses in the mapped left and right physical subdomains.
    res_minus = jax.vmap(lambda x, t: pde_residual(params["minus"], x, t))(x_minus, pde_t)
    res_plus = jax.vmap(lambda x, t: pde_residual(params["plus"], x, t))(x_plus, pde_t)
    loss_pde_minus = jnp.mean(res_minus**2)
    loss_pde_plus = jnp.mean(res_plus**2)
    loss_pde = loss_pde_minus + loss_pde_plus

    # Rankine-Hugoniot loss along x=n(t).
    def rh_residual(t):
        sx = shock_position(params["shock"], t)
        st = jax.grad(lambda tau: shock_position(params["shock"], tau))(t)
        u_minus = solution_net(params["minus"], sx, t)
        u_plus = solution_net(params["plus"], sx, t)
        return st * (u_plus - u_minus) - (flux(u_plus) - flux(u_minus))

    rh = jax.vmap(rh_residual)(rh_t)
    loss_rh = jnp.mean(rh**2)

    # Initial data on T_minus(xi,0) and T_plus(xi,0). Since n(0)=0, these
    # intervals are (-4,0) and (0,1), respectively.
    x0_minus = t_minus_x(0.0, ic_xi)
    x0_plus = t_plus_x(0.0, ic_xi)
    pred0_minus = jax.vmap(lambda x: solution_net(params["minus"], x, 0.0))(x0_minus)
    pred0_plus = jax.vmap(lambda x: solution_net(params["plus"], x, 0.0))(x0_plus)
    loss_ic_minus = jnp.mean((pred0_minus - u0(x0_minus)) ** 2)
    loss_ic_plus = jnp.mean((pred0_plus - u0(x0_plus)) ** 2)
    loss_ic = loss_ic_minus + loss_ic_plus

    total = LAMBDA_PDE * loss_pde + MU_RH * loss_rh + KAPPA_IC * loss_ic
    parts = {
        "loss": total,
        "pde": loss_pde,
        "pde_minus": loss_pde_minus,
        "pde_plus": loss_pde_plus,
        "rh": loss_rh,
        "ic": loss_ic,
        "ic_minus": loss_ic_minus,
        "ic_plus": loss_ic_plus,
    }
    return total, parts


def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, data):
        (loss_value, parts), grads = jax.value_and_grad(losses, has_aux=True)(params, data)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss_value, parts

    return train_step


def evaluate_solution(params, nx=400, nt=250):
    x = jnp.linspace(A, B, nx)
    t = jnp.linspace(0.0, T_FINAL, nt)
    shock_x = jax.vmap(lambda tt: shock_position(params["shock"], tt))(t)

    def eval_at_time(tt, sx):
        u_minus = jax.vmap(lambda xx: solution_net(params["minus"], xx, tt))(x)
        u_plus = jax.vmap(lambda xx: solution_net(params["plus"], xx, tt))(x)
        return jnp.where(x < sx, u_minus, u_plus)

    u = jax.vmap(eval_at_time)(t, shock_x)
    return np.asarray(x), np.asarray(t), np.asarray(u), np.asarray(shock_x)


def save_loss_history(path, history):
    fieldnames = [
        "epoch",
        "loss",
        "pde",
        "pde_minus",
        "pde_plus",
        "rh",
        "ic",
        "ic_minus",
        "ic_plus",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_outputs(outdir, x, t, u, shock_x):
    loss_df = pd.read_csv(os.path.join(outdir, "loss_history.csv"))

    plt.figure(figsize=(7, 4))
    for col in ["loss", "pde", "rh", "ic"]:
        plt.semilogy(loss_df["epoch"], loss_df[col], label=col)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    extent = [x.min(), x.max(), t.min(), t.max()]
    plt.imshow(u, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    plt.colorbar(label="u(x,t)")
    plt.plot(shock_x, t, "w-", linewidth=2.0, label="learned shock")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_solution_ndnn.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(t, shock_x, linewidth=2.0)
    plt.xlabel("t")
    plt.ylabel("n(t)")
    plt.title("Learned shock curve")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_shock_curve.png"), dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="One-shock NDNN Experiment 1")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--n-pde", type=int, default=2500)
    parser.add_argument("--n-rh", type=int, default=2500)
    parser.add_argument("--n-ic", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="runs/exp1")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    params = init_params(args.seed)
    data = make_training_points(args.seed + 1234, args.n_pde, args.n_rh, args.n_ic)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    train_step = make_train_step(optimizer)

    log_every = 500
    history = []

    print(f"Writing outputs to {args.outdir}")
    print(
        "Training with "
        f"epochs={args.epochs}, n_pde={args.n_pde}, n_rh={args.n_rh}, "
        f"n_ic={args.n_ic}, lr={args.lr}, seed={args.seed}"
    )

    for epoch in range(args.epochs + 1):
        params, opt_state, loss_value, parts = train_step(params, opt_state, data)

        if epoch % log_every == 0 or epoch == args.epochs:
            row = {"epoch": epoch}
            row.update({key: float(value) for key, value in parts.items()})
            history.append(row)
            print(
                f"epoch {epoch:7d} | loss={row['loss']:.6e} | "
                f"pde={row['pde']:.6e} | rh={row['rh']:.6e} | ic={row['ic']:.6e}"
            )

    save_loss_history(os.path.join(args.outdir, "loss_history.csv"), history)

    x, t, u, shock_x = evaluate_solution(params)
    np.savez(os.path.join(args.outdir, "solution_grid.npz"), x=x, t=t, u=u, shock_x=shock_x)

    shock_df = pd.DataFrame({"t": t, "n": shock_x})
    shock_df.to_csv(os.path.join(args.outdir, "shock_curve.csv"), index=False)

    plot_outputs(args.outdir, x, t, u, shock_x)
    print("Done.")


if __name__ == "__main__":
    main()
