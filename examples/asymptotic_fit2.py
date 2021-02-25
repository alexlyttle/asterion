"""
Asymptotic fit
"""
import jax
import jax.numpy as jnp
import numpy as np

from regression import init_optimizer, loss_fn, make_targets, get_update_fn, \
    make_plot
from parser import parse_args
from transforms import Bounded, Exponential, Union, Scale, Shift

a2 = Exponential()

def model(params, inputs):
    """
    parameter  name  shape  range        units transform
    =======================================================
    params[0]: a0    (N)    []
    params[1]: a1    (N)    []
    params[2]: a2    (N)    [1e-4, 1e-1] ---   exp

    inputs[0]: n     (N, M) [1, 40]      ---
    inputs[1]: Δν    (N)    [1e0, 1e3]   μHz
    inputs[2]: ν_max (N)    [1e1, 1e4]   μHz
    """
    _a2 = a2.forward(params[2])
    
    nu = (params[0] + params[1] * inputs[0] + _a2 * inputs[0]**2) * inputs[1]
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
    nu = star[3:]
    n = jnp.arange(nu.shape[0]) + 1

    idx_max = jnp.argmin(jnp.abs(nu - nu_max))
    n_max = n[idx_max]
    n_modes = 16
    
    idx = jnp.arange(idx_max - jnp.floor(n_modes/2), idx_max + jnp.ceil(n_modes/2), dtype=int)
    nu = nu[idx]
    n = n[idx]

    a0_init = 2.
    a1_init = 0.9
    a2_init = 5e-4

    params_init = (a0_init, a1_init, a2.inverse(a2_init))
    print('Initial parameters\n------------------')
    print(f'a0 = {a0_init:{fmt}}, a1 = {a1_init:{fmt}}, a2 = {a2_init:{fmt}}\n')

    inputs = (n, delta_nu)
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
    a0_fit, a1_fit, a2_fit = (*params_fit[0:2], a2.forward(params_fit[2]))
    print('Fit parameters\n--------------')
    print(f'a0 = {a0_fit:{fmt}}, a1 = {a1_fit:{fmt}}, a2 = {a2_fit:{fmt}}')

    if args.showplots:
        plot(params_init, params_fit, inputs, targets)


if __name__ == '__main__':
    main()
