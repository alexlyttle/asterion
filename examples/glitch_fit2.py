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
from jax.experimental import optimizers

a2 = Exponential()
a = Exponential()
b = Exponential()
tau = Exponential()
phi = Bounded(-jnp.pi, jnp.pi)

def asy_fit(n, delta_nu, a0, a1, a2):
    nu = (a0 + a1 * n + a2 * n**2) * delta_nu
    return nu

def he_amplitude(nu_asy, a, b):
    return a * nu_asy * jnp.exp(- b * nu_asy**2)

def he_glitch(nu_asy, a, b, tau, phi):
    return he_amplitude(nu_asy, a, b) * jnp.sin(4*jnp.pi*tau*nu_asy + phi)

def model(params, inputs):
    """
    parameter  name  shape  range        units transform
    =======================================================
    params[0]: a0    (N)    []
    params[1]: a1    (N)    []
    params[2]: a2    (N)    [1e-4, 1e-1] ---   exp   
    params[3]: a     (N)    [~ 1e-2]     μHz   exp
    params[4]: b     (N)    [~ 1e-6]     Ms2   exp
    params[5]: tau   (N)    [~ 1e-3]     Ms    exp
    params[6]: phi   (N)    [-pi, +pi]   ---   bounded

    inputs[0]: n     (N, M) [1, 40]      ---
    inputs[1]: Δν    (N)    [1e0, 1e3]   μHz
    inputs[2]: ν_max (N)    [1e1, 1e4]   μHz
    """
    _a2 = a2.forward(params[2])
    nu_asy = asy_fit(*inputs[:2], *params[:2], _a2)
    
    _a = a.forward(params[3])
    _b = b.forward(params[4])
    _tau = tau.forward(params[5])
    # _tau = inputs[2]**(-0.7)
    _phi = phi.forward(params[6])
    nu = nu_asy + he_glitch(nu_asy, _a, _b, _tau, _phi)

    return nu

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return jnp.array(data)

def plot(n, nu, delta_nu, nu_max, a0_fit, a1_fit, a2_fit, a_fit, b_fit, tau_fit, phi_fit):
    from matplotlib import pyplot as plt

    n_fit = jnp.linspace(n[0], n[-1], 200)

    nu_asy = asy_fit(n, delta_nu, a0_fit, a1_fit, a2_fit)
    nu_asy_fit = asy_fit(n_fit, delta_nu, a0_fit, a1_fit, a2_fit)

    dnu = nu - nu_asy
    # dnu_fit = he_glitch(nu_asy_fit, a_fit, b_fit, nu_max**(-0.7), phi_fit)
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
    nu = star[3:-1]
    n = jnp.arange(nu.shape[0]) + 1

    idx_max = jnp.argmin(jnp.abs(nu - nu_max))
    n_max = n[idx_max]
    n_modes = 16
    
    idx = jnp.arange(idx_max - jnp.floor(n_modes/2), idx_max + jnp.ceil(n_modes/2), dtype=int)
    nu = nu[idx]
    n = n[idx]

    a0_init, a1_init, a2_init = (1.7, 0.95, 8e-4)
    a_init, b_init = (1e-2, 1e-6)
    tau_init, phi_init = (1e-3, 0.0)

    params_init = (
        a0_init, a1_init, a2.inverse(a2_init),
        a.inverse(a_init), b.inverse(b_init),
        tau.inverse(tau_init), phi.inverse(phi_init),
    )

    print('Initial parameters\n------------------')
    print(f'a0  = {a0_init:{fmt}}, a1  = {a1_init:{fmt}}, a2  = {a2_init:{fmt}}')
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
    a0_fit, a1_fit, a2_fit = (*params_fit[:2], a2.forward(params_fit[2]))
    a_fit, b_fit = (a.forward(params_fit[2]), b.forward(params_fit[3]))
    tau_fit, phi_fit = (tau.forward(params_fit[4]), phi.forward(params_fit[5]))

    print('Fit parameters\n--------------')
    print(f'a0  = {a0_fit:{fmt}}, a1  = {a1_fit:{fmt}}, a2  = {a2_fit:{fmt}}')
    print(f'a   = {a_fit:{fmt}}, b   = {b_fit:{fmt}}')
    print(f'tau = {tau_fit:{fmt}}, phi = {phi_fit:{fmt}}')

    if args.showplots:
        plot(n, nu, delta_nu, nu_max, a0_fit, a1_fit, a2_fit, a_fit, b_fit, tau_fit, phi_fit)


if __name__ == '__main__':
    main()
