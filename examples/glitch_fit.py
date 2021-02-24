"""
Asymptotic fit
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
a = Exponential()
b = Exponential()
tau = Exponential()
phi = Bounded(-jnp.pi, jnp.pi)

def asy_fit(n, delta_nu, nu_max, epsilon, alpha):
    n_max = nu_max / delta_nu - epsilon
    nu = (n + epsilon + 0.5*alpha*(n - n_max)**2) * delta_nu
    return nu

def he_amplitude(nu_asy, a, b):
    return a * nu_asy * jnp.exp(- b * nu_asy**2)

def he_glitch(nu_asy, a, b, tau, phi):
    return he_amplitude(nu_asy, a, b) * jnp.sin(4*jnp.pi*tau*nu_asy + phi)

def model(params, inputs):
    """
    parameter  name  shape  range        units transform
    =======================================================
    params[0]: ε     (N)    [0.0, 2.0]   ---   bounded
    params[1]: α     (N)    [1e-4, 1]    ---   exp(bounded)
    params[2]: a     (N)    [~ 1e-2]     μHz   exp
    params[3]: b     (N)    [~ 1e-6]     Ms2   exp
    params[4]: tau   (N)    [~ 1e-3]     Ms    exp
    params[5]: phi   (N)    [-pi, +pi]   ---   bounded

    inputs[0]: n     (N, M) [1, 40]      ---
    inputs[1]: Δν    (N)    [1e0, 1e3]   μHz
    inputs[2]: ν_max (N)    [1e1, 1e4]   μHz
    """
    _epsilon = epsilon.forward(params[0])
    _alpha = alpha.forward(params[1])
    nu_asy = asy_fit(*inputs[:3], _epsilon, _alpha)
    
    _a = a.forward(params[2])
    _b = b.forward(params[3])
    _tau = tau.forward(params[4])
    _phi = phi.forward(params[5])
    nu = nu_asy + he_glitch(nu_asy, _a, _b, _tau, _phi)

    return nu

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return jnp.array(data)

def plot(n, nu, delta_nu, nu_max, eps_fit, alp_fit, a_fit, b_fit, tau_fit, phi_fit):
    from matplotlib import pyplot as plt

    n_fit = jnp.linspace(n[0], n[-1], 200)

    nu_asy = asy_fit(n, delta_nu, nu_max, eps_fit, alp_fit)
    nu_asy_fit = asy_fit(n_fit, delta_nu, nu_max, eps_fit, alp_fit)

    dnu = nu - nu_asy
    dnu_fit = he_glitch(nu_asy_fit, a_fit, b_fit, tau_fit, phi_fit)

    nu_fit = nu_asy_fit + dnu_fit

    fig, ax = plt.subplots()
    ax.plot(nu, dnu, '.')
    ax.plot(nu_fit, dnu_fit)
    ax.set_xlabel('ν (μHz)')
    ax.set_ylabel('δν (μHz)')

    plt.show()

def main():
    args = parse_args(__doc__, defaults={'l': 0.05, 'n': 1000, 'f': '.5f'})
    fmt = args.format

    data = load_data('data/modes.csv')
    star = data[data.shape[0]//2].flatten()
    delta_nu = star[1]
    nu_max = star[2]
    nu = star[3:]
    n = jnp.arange(nu.shape[0]) + 1

    idx_max = jnp.argmin(jnp.abs(nu - nu_max))
    n_max = n[idx_max]
    n_modes = 18
    
    idx = jnp.arange(idx_max - jnp.floor(n_modes/2), idx_max + jnp.ceil(n_modes/2), dtype=int)
    nu = nu[idx]
    n = n[idx]

    eps_init, alp_init = (1.5, 1e-3)
    a_init, b_init = (1e-2, 1e-6)
    tau_init, phi_init = (1e-3, 0.0)

    params_init = (
        epsilon.inverse(eps_init), alpha.inverse(alp_init),
        a.inverse(a_init), b.inverse(b_init),
        tau.inverse(tau_init), phi.inverse(phi_init),
    )

    print('Initial parameters\n------------------')
    print(f'ε   = {eps_init:{fmt}}, α   = {alp_init:{fmt}}')
    print(f'a   = {a_init:{fmt}}, b   = {b_init:{fmt}}')
    print(f'tau = {tau_init:{fmt}}, phi = {phi_init:{fmt}}\n')

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
    a_fit, b_fit = (a.forward(params_fit[2]), b.forward(params_fit[3]))
    tau_fit, phi_fit = (tau.forward(params_fit[4]), phi.forward(params_fit[5]))

    print('Fit parameters\n--------------')
    print(f'ε   = {eps_fit:{fmt}}, α   = {alp_fit:{fmt}}')
    print(f'a   = {a_fit:{fmt}}, b   = {b_fit:{fmt}}')
    print(f'tau = {tau_fit:{fmt}}, phi = {phi_fit:{fmt}}')

    if args.showplots:
        plot(n, nu, delta_nu, nu_max, eps_fit, alp_fit, a_fit, b_fit, tau_fit, phi_fit)


if __name__ == '__main__':
    main()
