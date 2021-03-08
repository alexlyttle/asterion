"""
Asymptotic fit

Fits a parametrised asymptotic relation to the radial oscillation modes to 2nd
order in n.
"""
import jax
import jax.numpy as jnp
import numpy as np

from regression import init_optimizer, loss_fn, make_targets, get_update_fn, \
    make_plot
from parser import parse_args
from transforms import Bounded, Exponential, Union

epsilon = Bounded(0., 2.)
alpha = Union(Bounded(jnp.log(1e-4), jnp.log(1)), Exponential())

def model(params, inputs):
    """
    parameter  name  shape  range        units transform
    =======================================================
    params[0]: ε     (N)    [0.0, 2.0]   ---   bounded
    params[1]: α     (N)    [1e-4, 1]    ---   exp(bounded)

    inputs[0]: n     (N, M) [1, 40]      ---
    inputs[1]: Δν    (N)    [1e0, 1e3]   μHz
    inputs[2]: ν_max (N)    [1e1, 1e4]   μHz
    """
    _epsilon = epsilon.forward(params[0])
    _alpha = alpha.forward(params[1])
    n_max = inputs[2] / inputs[1] - _epsilon
    nu = (inputs[0] + _epsilon + 0.5*_alpha*(inputs[0] - n_max)**2) * inputs[1]
    return nu

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return jnp.array(data)

def plot(params_init, params_fit, inputs, targets):
    from matplotlib import pyplot as plt

    n = inputs[0]
    delta_nu = inputs[1]
    nu = targets

    nu_init = model(params_init, inputs)
    nu_fit = model(params_fit, inputs)

    fig, ax = plt.subplots()
    ax = make_plot(ax, n, nu, y_init=nu_init, y_fit=nu_fit)
    ax.set_xlabel('n')
    ax.set_ylabel('ν (μHz)')

    fig, ax = plt.subplots()
    ax = make_plot(ax, nu, nu%delta_nu, y_init=nu_init%delta_nu, 
                    y_fit=nu_fit%delta_nu)
    ax.set_xlabel('ν (μHz)')
    ax.set_ylabel('ν mod. Δν (μHz)')

    plt.show()

def main():
    args = parse_args(__doc__, defaults={'l': 0.1, 'n': 1000, 'f': '.4f'})
    fmt = args.format

    data = load_data('data/modes.csv')
    star = data[data.shape[0]//2].flatten()
    delta_nu = star[1]
    nu_max = star[2]
    nu = star[3:-1]
    n = jnp.arange(nu.shape[0]) + 1

    idx_max = jnp.argmin(jnp.abs(nu - nu_max))
    n_max = n[idx_max]
    n_modes = 16
    
    idx = jnp.arange(idx_max - jnp.floor(n_modes/2), idx_max + jnp.ceil(n_modes/2), dtype=int)
    nu = nu[idx]
    n = n[idx]

    eps_init = 1.5
    alp_init = 1e-3

    params_init = (epsilon.inverse(eps_init), alpha.inverse(alp_init))
    print('Initial parameters\n------------------')
    print(f'ε = {eps_init:{fmt}}, α = {alp_init:{fmt}}\n')

    inputs = (n, delta_nu, nu_max)
    targets = nu

    opt_state, opt_update, get_params = init_optimizer(params_init, args.lrate)
    update = get_update_fn(opt_update, get_params, inputs, targets, model)
    
    print('Fitting\n-------')
    for i in range(args.numsteps):
        value, opt_state = update(i, opt_state)
        if args.verbose:
            print(f'loss = {value:{fmt}}') 
    print(f'mean squared error = {value:{fmt}}\n')

    params_fit = get_params(opt_state)
    eps_fit, alp_fit = (epsilon.forward(params_fit[0]), alpha.forward(params_fit[1]))
    print('Fit parameters\n--------------')
    print(f'ε = {eps_fit:{fmt}}, α = {alp_fit:{fmt}}')

    if args.showplots:
        plot(params_init, params_fit, inputs, targets)


if __name__ == '__main__':
    main()
