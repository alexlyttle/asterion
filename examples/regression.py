"""
regression.py
"""
import jax
import jax.numpy as jnp
import logging

from jax.experimental import optimizers

logging.getLogger('absl').setLevel('ERROR')

def loss_fn(params, inputs, targets, model):
    prediction = model(params, inputs)
    return jnp.mean((targets - prediction)**2)

def make_plot(ax, x, y_obs, y_true=None, y_fit=None, y_init=None):
    ax.plot(x, y_obs, '.', label='obs')
    if y_init is not None:
        ax.plot(x, y_init, label=f'init')
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
