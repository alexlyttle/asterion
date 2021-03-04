"""
Asymptotic fit
"""
import jax
import jax.numpy as jnp
import numpy as np

from regression import init_optimizer, make_targets, get_update_fn, make_plot
from parser import parse_args
from transforms import Bounded, Exponential, Union
from jax.experimental import optimizers
from jax.scipy.optimize import minimize

from scipy.optimize import curve_fit

b0 = Exponential()
b1 = Union(Bounded(jnp.log(1e-7), jnp.log(1e-5)), Exponential())
tau = Union(Bounded(jnp.log(1e-4), jnp.log(1e-2)), Exponential())
phi = Bounded(-jnp.pi, jnp.pi)

def asy_fit(n, a0, a1, a2, a3, a4):
    nu = a0 + a1 * n + a2 * n**2 + a3 * n**3 + a4 * n**4
    return nu

def get_loss_fn(inputs, targets, reg=0.0):
    def loss_fn(params):
        prediction = asy_fit(inputs, *params)
        loss = jnp.sum((targets - prediction)**2) + reg**2 * jnp.sum((params[3] + params[4]*inputs)**2)
        return loss
    return loss_fn

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return jnp.array(data)

def plot(n, nu, delta_nu, a0, a1, a2, a3, a4):
    from matplotlib import pyplot as plt

    # n_fit = jnp.linspace(n[0], n[-1], 200)

    nu_asy = asy_fit(n, a0, a1, a2, a3, a4)
    # nu_asy_fit = asy_fit(n_fit, a0, a1, a2, a3, a4)

    dnu = nu - nu_asy

    fig, ax = plt.subplots()
    ax.plot(nu, nu%delta_nu, '.')
    ax.plot(nu_asy, nu_asy%delta_nu)
    ax.set_xlabel('ν (μHz)')
    ax.set_ylabel('ν mod. Δν (μHz)')

    fig, ax = plt.subplots()
    ax.plot(nu_asy, dnu, '.')
    ax.set_xlabel('ν (μHz)')
    ax.set_ylabel('δν (μHz)')

    plt.show()

def main():
    args = parse_args(__doc__, defaults={'l': 0.05, 'n': 1000, 'f': '.3e'})
    fmt = args.format

    data = load_data('data/modes.csv')
    star = data[data.shape[0]//2].flatten()
    delta_nu = star[1]
    nu_max = star[2]
    nu = star[3:-1]
    n = jnp.arange(nu.shape[0]) + 1

    idx_max = jnp.argmin(jnp.abs(nu - nu_max))
    n_max = n[idx_max]
    n_modes = 24
    
    idx = jnp.arange(idx_max - jnp.floor(n_modes/2), idx_max + jnp.ceil(n_modes/2), dtype=int)
    nu = nu[idx]
    n = n[idx]

    a0_init, a1_init, a2_init, a3_init, a4_init = (1.4*delta_nu, delta_nu, 1e-1, 0.0, 0.0)

    params_init = jnp.array([
        a0_init, a1_init, a2_init, a3_init, a4_init,
    ])

    print('Initial parameters\n------------------')
    print(f'a0  = {a0_init:{fmt}}, a1  = {a1_init:{fmt}}, a2  = {a2_init:{fmt}}')
    print(f'a3  = {a3_init:{fmt}}, a4  = {a4_init:{fmt}}\n')

    loss = get_loss_fn(n, nu, reg=1e2)
    opt_results = minimize(loss, params_init, method='BFGS')
    print(opt_results.x)
    # print('Fit parameters\n--------------')
    # print(f'a0  = {a0_fit:{fmt}}, a1  = {a1_fit:{fmt}}, a2  = {a2_fit:{fmt}}')
    # print(f'a3  = {a3_fit:{fmt}}, a4  = {a4_fit:{fmt}}')

    if args.showplots:
        plot(n, nu, delta_nu, *opt_results.x)


if __name__ == '__main__':
    main()
