"""
Asymptotic fit
"""
import jax
import jax.numpy as jnp
import numpy as np

from regression import init_optimizer, loss_fn, make_targets, get_update_fn, \
    make_plot
from parser import parse_args

def bounded(low, high):
    """
    Returns a function which bounds an input between `low` and `high`.
    """
    return lambda x: low + (high - low) * jax.nn.sigmoid(x)

def unbounded(low, high):
    """
    Returns a function which unbounds an input between `low` and `high`.
    """
    return lambda y: jnp.log(y - low) - jnp.log(high - y)

def exp(x):
    return jnp.exp(x)

def log(y):
    return jnp.log(y)

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
    eps = bounded(0., 2.0)(params[0])
    alpha =  exp(bounded(log(1e-4), log(1))(params[1]))
    n_max = inputs[2] / inputs[1] - eps

    a0 = eps + 0.5*alpha*n_max**2
    a1 = 1 - alpha*n_max
    a2 = 0.5*alpha

    nu = (a0 + a1 * inputs[0] + a2 * inputs[0]**2) * inputs[1]
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
    ax.set_ylabel('nu (μHz)')

    fig, ax = plt.subplots()
    ax = make_plot(ax, nu, nu%delta_nu, y_init=nu_init%delta_nu, 
                    y_fit=nu_fit%delta_nu)
    ax.set_xlabel('nu (μHz)')
    ax.set_ylabel('nu mod. delta_nu (μHz)')

    plt.show()

def main():
    args = parse_args(__doc__, defaults={'l': 0.0001, 'n': 10000})
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

    eps_init = 1.5
    alpha_init = 1e-3

    params_init = (unbounded(0., 2.0)(1.5), unbounded(log(1e-4), log(1))(log(5e-4)))
    print('Initial parameters\n------------------')
    print(params_init)

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
    print('Fit parameters\n--------------')
    print(params_fit)   

    if args.showplots:
        plot(params_init, params_fit, inputs, targets)


if __name__ == '__main__':
    main()
