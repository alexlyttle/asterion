"""
Asymptotic fit 3

Fits a 4th order polynomial to the radial oscillation modes.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

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

def asy_fit(n, delta_nu, n_max, epsilon, alpha, a3, a4):
    # n_max = nu_max / delta_nu - epsilon
    # nu = a0 + a1 * n + a2 * n**2 + a3 * n**3 + a4 * n**4
    nu = delta_nu * (n + epsilon + 0.5*alpha*(n_max - n)**2 + a3 * n**3 + a4 * (n_max - n)**4)
    return nu

def model(params, inputs):
    return asy_fit(inputs, *params)

# def get_loss_fn(inputs, targets, reg=0.0):
#     def loss_fn(params):
#         prediction = asy_fit(inputs, *params)
#         loss = jnp.sum((targets - prediction)**2) + reg**2 * jnp.sum((params[-2] + params[-1]*inputs)**2)
#         return loss
#     return loss_fn

def get_loss_fn(reg=0.0):
    def loss_fn(params, inputs, targets, model):
        prediction = model(params, inputs)
        # n_max = params[1] / params[0] - params[2]
        n_max = params[1]
        loss = jnp.mean((targets - prediction)**2) + reg**2 * jnp.mean((params[0] * (6*params[-2] + 24*params[-1]*(inputs - n_max)))**2)
        return loss
    return loss_fn

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return jnp.array(data)

def plot(n, nu, delta_nu, nu_max, epsilon, alpha, a3, a4):

    # n_fit = jnp.linspace(n[0], n[-1], 200)

    nu_asy = asy_fit(n, delta_nu, nu_max, epsilon, alpha, a3, a4)
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

    args = parse_args(__doc__, defaults={'l': 0.01, 'n': 1000, 'f': '.3e'})
    fmt = args.format

    data = load_data('data/modes.csv')
    star = data[data.shape[0]//2].flatten()
    delta_nu = star[1]
    nu_max = star[2]
    nu = star[3:-1]
    n = jnp.arange(nu.shape[0]) + 1

    fig, ax = plt.subplots()
    ax.plot(nu, nu%delta_nu, '.')
    ax.set_xlabel('ν (μHz)')
    ax.set_ylabel('ν mod. Δν (μHz)')

    idx_max = jnp.argmin(jnp.abs(nu - nu_max))
    n_max = n[idx_max]
    n_modes = 34
    
    idx = jnp.arange(idx_max - jnp.floor(n_modes/2), idx_max + jnp.ceil(n_modes/2), dtype=int)
    nu = nu[idx]
    n = n[idx]

    epsilon_init, alpha_init = (1.3, 1e-3)
    a3_init, a4_init = (1e-6, -6e-6)

    n_max_init = nu_max / delta_nu + epsilon_init

    params_init = jnp.array([
        delta_nu, n_max_init, epsilon_init, alpha_init, a3_init, a4_init,
    ])

    print('Initial parameters\n------------------')
    print(f'delta_nu  = {delta_nu:{fmt}}, nu_max  = {nu_max:{fmt}}, eps  = {epsilon_init:{fmt}}')
    print(f'alpha = {alpha_init:{fmt}}, a3  = {a3_init:{fmt}}, a4  = {a4_init:{fmt}}\n')

    loss = get_loss_fn(reg=1e-10)
    # opt_results = minimize(loss, params_init, method='BFGS')
    # print(opt_results.x)

    opt_state, opt_update, get_params = init_optimizer(params_init, args.lrate)
    update = get_update_fn(opt_update, get_params, n, nu, model, loss=loss)
    
    print('Fitting\n-------')
    for i in range(args.numsteps):
        value, opt_state = update(i, opt_state)
        if args.verbose:
            print(f'loss = {value:{fmt}}') 
    print(f'mean squared error = {value:{fmt}}\n')

    params_fit = get_params(opt_state)

    print('Fit parameters\n--------------')
    print(f'{params_fit}')

    if args.showplots:
        plot(n, nu, *params_fit)


if __name__ == '__main__':
    main()
