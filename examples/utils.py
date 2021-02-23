"""
utils.py
"""
import jax
import jax.numpy as jnp

from jax.experimental import optimizers
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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

def loss_fn(params, inputs, targets, model):
    prediction = model(params, inputs)
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

def make_targets(key, params, inputs, model, scale=0.1):
    targets = model(params, inputs)
    targets = targets + scale * jax.random.normal(key, targets.shape)
    return targets

def init_optimizer(params, learning_rate):
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_state = opt_init(params) 
    return opt_state, opt_update, get_params

def get_update_fn(opt_update, get_params, inputs, targets, model):

    @jax.jit
    def update(i, opt_state):
        params = get_params(opt_state)
        value, grads = jax.value_and_grad(loss_fn)(params, inputs, targets, model)
        opt_state = opt_update(i, grads, opt_state)
        return value, opt_state
    
    return update
