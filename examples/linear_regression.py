"""
Basic linear model.
"""
import jax
import jax.numpy as jnp

from jax.experimental import optimizers
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial

def parse_args():
    parser = ArgumentParser(
        description=__doc__, 
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-e', '--error', metavar='ERR', type=float,
                        default=0.1, help='error (noise) applied to targets')
    parser.add_argument('-f', '--format', metavar='FMT', type=str,
                        default='.3f', help='output float format ')
    parser.add_argument('-l', '--lrate', metavar='LRT', type=float,
                        default=0.1, help='learning rate')
    parser.add_argument('-n', '--numsteps', metavar='N', type=int,
                        default=100, help='number of steps to train')
    parser.add_argument('-p', '--showplots', action='store_true',
                        help='show plots (requires matplotlib)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose output')
    args = parser.parse_args()
    return args


def predict(params, inputs):
    prediction = params[0] + jnp.dot(params[1], inputs)
    return prediction

def loss_fn(params, inputs, targets):
    prediction = predict(params, inputs)
    return jnp.mean((targets - prediction)**2)

def make_plot(ax, x, y_obs, y_true=None, y_fit=None):
    ax.plot(x, y_obs, '.', label='obs')
    if y_true is not None:
        ax.plot(x, y_true, label=f'true')
    if y_fit is not None:
        ax.plot(x, y_fit, label=f'fit')
    ax.set_xlabel('input')
    ax.set_ylabel('targets')
    ax.legend()
    return ax

def make_targets(key, params, inputs, scale=0.1):
    targets = predict(params, inputs)
    targets = targets + scale * jax.random.normal(key, inputs.shape)
    return targets

def init_optimizer(params, learning_rate):
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_state = opt_init(params) 
    return opt_state, opt_update, get_params

def get_update_fn(opt_update, get_params, inputs, targets):

    @jax.jit
    def update(i, opt_state):
        params = get_params(opt_state)
        value, grads = jax.value_and_grad(loss_fn)(params, inputs, targets)
        opt_state = opt_update(i, grads, opt_state)
        return value, opt_state
    
    return update

def main():
    import logging
    logging.getLogger('absl').setLevel('ERROR')

    args = parse_args()
    fmt = args.format

    a_true, b_true = params_true = (-5., 2.)
    print('True parameters\n---------------')
    print(f'a_true = {a_true:{fmt}}, b_true = {b_true:{fmt}}\n')

    n_data = 100
    x = jnp.linspace(0, 1, n_data)
    y_obs = make_targets(jax.random.PRNGKey(42),
                         params_true, x, scale=args.error)

    a_init, b_init = init_params = (-3., 1.)
    print('Initial parameters\n------------------')
    print(f'a_init = {a_init:{fmt}}, b_init = {b_init:{fmt}}\n')   

    opt_state, opt_update, get_params = init_optimizer(init_params, args.lrate)
    update = get_update_fn(opt_update, get_params, x, y_obs)
    
    print('Fitting\n-------')
    for i in range(args.numsteps):
        value, opt_state = update(i, opt_state)
        if args.verbose:
            print(f'loss = {value:{fmt}}') 
    print(f'mean squared error = {value:{fmt}}\n')

    a_fit, b_fit = params_fit = get_params(opt_state)
    print('Fit parameters\n--------------')
    print(f'a_fit  = {a_fit:{fmt}}, b_fit  = {b_fit:{fmt}}')   

    if args.showplots:
        from matplotlib import pyplot as plt

        y_true = predict(params_true, x)
        y_fit = predict(params_fit, x)
        
        fig, ax = plt.subplots()
        ax = make_plot(ax, x, y_obs, y_true, y_fit)

        plt.show()


if __name__ == '__main__':
    main()
