"""
Glitch fit 2

Fits a helium-II ionisation zone glitch to the radial oscillation modes
assuming a background governed by a 4th order polynomial in n.
"""
import jax
import jax.numpy as jnp
import numpy as np

from regression import init_optimizer, make_targets, get_update_fn, make_plot
from parser import parse_args
from transforms import Bounded, Exponential, Union
from jax.experimental import optimizers

b0 = Exponential()
b1 = Union(Bounded(jnp.log(1e-7), jnp.log(1e-5)), Exponential())
tau = Union(Bounded(jnp.log(1e-4), jnp.log(1e-2)), Exponential())
phi = Bounded(-jnp.pi, jnp.pi)

def loss_fn(params, inputs, targets, model):
    prediction = model(params, inputs)
    loss = jnp.mean((targets - prediction)**2) + inputs[-1]**2 * jnp.sum((params[3] + params[4]*inputs[0])**2)
    return loss

def asy_fit(n, a0, a1, a2, a3, a4):
    nu = a0 + a1 * n + a2 * n**2 + a3 * n**3 + a4 * n**4
    return nu

def he_amplitude(nu_asy, b0, b1):
    return b0 * nu_asy * jnp.exp(- b1 * nu_asy**2)

def he_glitch(nu_asy, b0, b1, tau, phi):
    return he_amplitude(nu_asy, b0, b1) * jnp.sin(4*jnp.pi*tau*nu_asy + phi)

def model(params, inputs):
    """
    parameter  name  shape  range        units transform
    =======================================================
    params[0]: a0    (N)    []
    params[1]: a1    (N)    []
    params[2]: a2    (N)    [1e-4, 1e-1] ---   exp   
    params[3]: a3    (N)    [1e-4, 1e-1] ---   exp   
    params[4]: a4    (N)    [1e-4, 1e-1] ---   exp   
    params[5]: b0     (N)    [~ 1e-2]     μHz   exp
    params[6]: b1     (N)    [~ 1e-6]     Ms2   exp
    params[7]: tau   (N)    [~ 1e-3]     Ms    exp
    params[8]: phi   (N)    [-pi, +pi]   ---   bounded

    inputs[0]: n     (N, M) [1, 40]      ---
    inputs[1]: Δν    (N)    [1e0, 1e3]   μHz
    inputs[2]: ν_max (N)    [1e1, 1e4]   μHz
    """
    nu_asy = asy_fit(inputs[0], *params[:5])
    
    _b0 = b0.forward(params[5])
    _b1 = b1.forward(params[6])
    _tau = tau.forward(params[7])
    _phi = phi.forward(params[8])
    # nu = nu_asy + he_glitch(nu_asy, _b0, _b1, _tau, _phi)
    nu = nu_asy + he_glitch(inputs[0]*inputs[1], _b0, _b1, _tau, _phi)
    return nu

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return jnp.array(data)

def plot(n, nu, delta_nu, a0, a1, a2, a3, a4, b0, b1, tau, phi):
    from matplotlib import pyplot as plt

    n_fit = jnp.linspace(n[0], n[-1], 200)

    nu_asy = asy_fit(n, a0, a1, a2, a3, a4)
    nu_asy_fit = asy_fit(n_fit, a0, a1, a2, a3, a4)

    dnu = nu - nu_asy
    # dnu_fit = he_glitch(nu_asy_fit, b0, b1, tau, phi)
    dnu_fit = he_glitch(n_fit*delta_nu, b0, b1, tau, phi)
    nu_fit = nu_asy_fit + dnu_fit

    fig, ax = plt.subplots()
    ax.plot(nu, nu%delta_nu, '.')
    ax.plot(nu_asy, nu_asy%delta_nu)
    ax.set_xlabel('ν (μHz)')
    ax.set_ylabel('ν mod. Δν (μHz)')

    fig, ax = plt.subplots()
    ax.plot(nu, dnu, '.')
    ax.plot(nu_fit, dnu_fit)
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
    b0_init, b1_init = (1e-2, 1e-6)
    tau_init, phi_init = (nu_max**(-0.9), 0.0)

    params_init = (
        a0_init, a1_init, a2_init, a3_init, a4_init,
        b0.inverse(b0_init), b1.inverse(b1_init),
        tau.inverse(tau_init), phi.inverse(phi_init),
    )

    print('Initial parameters\n------------------')
    print(f'a0  = {a0_init:{fmt}}, a1  = {a1_init:{fmt}}, a2  = {a2_init:{fmt}}')
    print(f'a3  = {a3_init:{fmt}}, a4  = {a4_init:{fmt}}')
    print(f'b0  = {b0_init:{fmt}}, b1  = {b1_init:{fmt}}')
    print(f'tau = {tau_init:{fmt}}, phi = {phi_init:{fmt}}\n')

    reg = 0.0001
    inputs = (n, delta_nu, reg)
    targets = nu

    # if args.showplots:
        # plot(n, nu, delta_nu, a0_init, a1_init, a2_init, a3_init, a4_init, b0_init, b1_init, tau_init, phi_init)


    opt_state, opt_update, get_params = init_optimizer(params_init, args.lrate)
    update = get_update_fn(opt_update, get_params, inputs, targets, model, loss=loss_fn)
    
    print('Fitting\n-------')
    for i in range(args.numsteps):
        value, opt_state = update(i, opt_state)
        if args.verbose:
            print(f'loss = {value:{fmt}}') 
    print(f'mean squared error = {value:{fmt}}\n')

    params_fit = get_params(opt_state)
    a0_fit, a1_fit, a2_fit, a3_fit, a4_fit = params_fit[:5]
    b0_fit, b1_fit = (b0.forward(params_fit[5]), b1.forward(params_fit[6]))
    tau_fit, phi_fit = (tau.forward(params_fit[7]), phi.forward(params_fit[8]))

    print('Fit parameters\n--------------')
    print(f'a0  = {a0_fit:{fmt}}, a1  = {a1_fit:{fmt}}, a2  = {a2_fit:{fmt}}')
    print(f'a3  = {a3_fit:{fmt}}, a4  = {a4_fit:{fmt}}')
    print(f'b0   = {b0_fit:{fmt}}, b1   = {b1_fit:{fmt}}')
    print(f'tau = {tau_fit:{fmt}}, phi = {phi_fit:{fmt}}')

    if args.showplots:
        plot(n, nu, delta_nu, a0_fit, a1_fit, a2_fit, a3_fit, a4_fit, b0_fit, b1_fit, tau_fit, phi_fit)


if __name__ == '__main__':
    main()
