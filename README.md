# NDNN Experiment 1

This repo contains a small Python/JAX version of Experiment 1 from
"Non-diffusive neural network method for hyperbolic conservation laws."  The
focus is the one-shock setup from Section 2.1.

The script solves

```text
u_t + (f(u))_x = 0,       f(u) = 4u(2-u)
```

on `(-4, 1) x [0, 3/4]`, using the three-state initial condition from the
paper.  There are two solution networks, `N_minus` and `N_plus`, with a learned
shock curve `n(t)` between them.  The shock network is written so that
`n(0)=0`.

## Initial Condition

The initial data is piecewise constant:

```text
u(x,0) = 1.0,   x < -2
u(x,0) = 0.5,  -2 <= x < 0
u(x,0) = 1.5,   x >= 0
```

The jump at `x=-2` creates the rarefaction, and the jump at `x=0` is the shock
tracked by the NDNN model.
