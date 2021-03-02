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
from matplotlib import pyplot as plt

epsilon = Bounded(0., 2.)
alpha = Union(Bounded(jnp.log(1e-4), jnp.log(1)), Exponential())
a = Exponential()
b = Exponential()
# tau = Exponential()
m = Exponential()
phi = Bounded(-jnp.pi, jnp.pi)

def asy_fit(n, delta_nu, nu_max, epsilon, alpha):
    n_max = nu_max / delta_nu - epsilon
    nu = (n + epsilon[..., jnp.newaxis] + 0.5*alpha[..., jnp.newaxis]*(n - n_max[..., jnp.newaxis])**2) * delta_nu[..., jnp.newaxis]
    return nu

def he_amplitude(nu_asy, a, b):
    return a[..., jnp.newaxis] * nu_asy * jnp.exp(- b[..., jnp.newaxis] * nu_asy**2)

def he_glitch(nu_asy, a, b, tau, phi):
    return he_amplitude(nu_asy, a, b) * jnp.sin(4*jnp.pi*tau[..., jnp.newaxis]*nu_asy + phi[..., jnp.newaxis])

def model(params, inputs):
    """
    parameter  name  shape  range        units transform
    =======================================================
    params[0]: ε     (N)    [0.0, 2.0]   ---   bounded
    params[1]: α     (N)    [1e-4, 1]    ---   exp(bounded)
    params[2]: a     (N)    [~ 1e-2]     μHz   exp
    params[3]: b     (N)    [~ 1e-6]     Ms2   exp
    params[4]: m     (N)    [~ 1]        ---   exp
    params[5]: phi   (N)    [-pi, +pi]   ---   bounded
    params[6]: d
    params[7]: dnu
    params[8]: numax

    inputs: n     (N, M) [1, 40]      ---
    # inputs[1]: Δν    (N)    [1e0, 1e3]   μHz
    # inputs[2]: ν_max (N)    [1e1, 1e4]   μHz
    """
    _epsilon = epsilon.forward(params[0])
    _alpha = alpha.forward(params[1])
    nu_asy = asy_fit(inputs, *params[7:], _epsilon, _alpha)
    
    _a = a.forward(params[2])
    _b = b.forward(params[3])
    # _tau = tau.forward(params[4])
    _tau = params[6] * params[8]**(- m.forward(params[4]))

    _phi = phi.forward(params[5])
    nu = nu_asy + he_glitch(nu_asy, _a, _b, _tau, _phi)

    return nu

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    return jnp.array(data)

def plot(n, nu, delta_nu, nu_max, eps_fit, alp_fit, a_fit, b_fit, m_fit, phi_fit, d_fit):

    n_fit = jnp.linspace(n[0], n[-1], 200)
    tau_fit = d_fit * nu_max**(- m_fit)

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

    # plt.show() 

def main():
    args = parse_args(__doc__, defaults={'l': 0.01, 'n': 1000, 'f': '.5f'})
    fmt = args.format

    data = load_data('data/modes.csv')
    # star = data[data.shape[0]//2].flatten()
    key = jax.random.PRNGKey(42)

    n_stars = 100
    star = data[jax.random.randint(key, (n_stars,), 0, data.shape[0])]
    delta_nu = star[..., 1]
    nu_max = star[..., 2]
    nu = star[..., 3:]
    n = jnp.ones(nu.shape) * jnp.arange(nu.shape[-1]) + 1
    helium = star[..., -1]

    idx_max = jnp.argmin(jnp.abs(nu - nu_max[..., jnp.newaxis]), axis=-1)
    n_max = jnp.array([n[i, idx] for i, idx in enumerate(idx_max)])
    n_modes = 16
    
    idx = jnp.linspace(
        idx_max - jnp.floor(n_modes/2), idx_max + jnp.ceil(n_modes/2), 
        n_modes, axis=-1, dtype=int
    )
    nu = jnp.array([nu[i, id] for i, id in enumerate(idx)])
    n = jnp.array([n[i, id] for i, id in enumerate(idx)])

    _param = jnp.ones(nu.shape[0])
    # eps_init = 1.4 * _param
    eps_init = 0.3 * jnp.log10(nu_max) + 0.5
    # alp_init = 1e-3 * _param
    alp_init = 1/nu_max
    a_init = 1e-2 * _param
    b_init = 1e-6 * _param
    # tau_init = 1e-3 * _param
    m_init = 0.9
    phi_init = 0. * _param
    d_init = 1.
    delta_nu_init = delta_nu
    nu_max_init = nu_max


    params_init = (
        epsilon.inverse(eps_init), alpha.inverse(alp_init),
        a.inverse(a_init), b.inverse(b_init),
        # tau.inverse(tau_init), phi.inverse(phi_init),
        m.inverse(m_init), phi.inverse(phi_init),
        d_init, delta_nu_init, nu_max_init
    )

    for j in range(3):
        # j = 0
        print('Initial parameters\n------------------')
        print(f'ε   = {eps_init[j]:{fmt}}, α   = {alp_init[j]:{fmt}}')
        print(f'a   = {a_init[j]:{fmt}}, b   = {b_init[j]:{fmt}}')
        # print(f'tau = {tau_init[j]:{fmt}}, phi = {phi_init[j]:{fmt}}\n')
        print(f'delta_nu = {delta_nu_init[j]:{fmt}}, nu_max = {nu_max_init[j]:{fmt}}')
        print(f'm   = {m_init:{fmt}}, phi = {phi_init[j]:{fmt}}')
        print(f'd   = {d_init:{fmt}}\n')

    # inputs = (n, delta_nu, nu_max)
    inputs = n
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
    # tau_fit, phi_fit = (tau.forward(params_fit[4]), phi.forward(params_fit[5]))
    m_fit, phi_fit = (m.forward(params_fit[4]), phi.forward(params_fit[5]))
    d_fit, delta_nu_fit, nu_max_fit = params_fit[6:]

    for j in range(3):
        print('Fit parameters\n--------------')
        print(f'ε   = {eps_fit[j]:{fmt}}, α   = {alp_fit[j]:{fmt}}')
        print(f'a   = {a_fit[j]:{fmt}}, b   = {b_fit[j]:{fmt}}')
        # print(f'tau = {tau_fit[j]:{fmt}}, phi = {phi_fit[j]:{fmt}}')
        print(f'delta_nu = {delta_nu_fit[j]:{fmt}}, nu_max = {nu_max_fit[j]:{fmt}}')
        print(f'm   = {m_fit:{fmt}}, phi = {phi_fit[j]:{fmt}}')
        print(f'd   = {d_fit:{fmt}}')
    
    amp = a_fit * nu_max * jnp.exp(- b_fit * nu_max**2)
    fig, ax = plt.subplots()
    ax.plot(helium, amp, 'o')
    # plt.show()
    # print(amp)
    if args.showplots:
        # plot(n[j], nu[j], delta_nu[j], nu_max[j], eps_fit[j], alp_fit[j], a_fit[j], b_fit[j], tau_fit[j], phi_fit[j])
        for j in range(10):
            plot(n[j], nu[j], delta_nu_fit[j], nu_max_fit[j], eps_fit[j], alp_fit[j], a_fit[j], b_fit[j], m_fit, phi_fit[j], d_fit)

        plt.show()

if __name__ == '__main__':
    main()
