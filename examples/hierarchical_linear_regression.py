"""
Hierarchical linear regression.
"""
import jax
import jax.numpy as jnp

from utils import parse_args, init_optimizer, loss_fn, make_targets, \
    get_update_fn, make_plot

def model(params, inputs):
    b = inputs[0]**params[1]  # b is related to beta via power law
    prediction = params[0] + b * inputs[1]
    return prediction

def main():
    args = parse_args(__doc__)
    fmt = args.format

    n_obj = 100
    
    a_true = jnp.linspace(0, 1, n_obj)
    b_true = -1.0
    params_true = (a_true, b_true)
    print('True parameters\n---------------')
    print(f'a_true =\n{a_true},\nb_true = {b_true:{fmt}}\n')

    n_data = 10
    x = jnp.linspace(0, 1, n_data)
    x = jnp.stack([x for _ in range(n_obj)], axis=1)
    print(x.shape)
    beta = jnp.linspace(1, 2, n_obj)
    key = jax.random.PRNGKey(42)
    y_obs = make_targets(key, params_true, (beta, x), model, scale=args.error)

    a_init, b_init = params_init = (
        jax.random.uniform(key, (n_obj,)),
        0.0
    )
    print('Initial parameters\n------------------')
    print(f'a_init =\n{a_init},\nb_init = {b_init:{fmt}}\n')   

    opt_state, opt_update, get_params = init_optimizer(params_init, args.lrate)
    update = get_update_fn(opt_update, get_params, (beta, x), y_obs, model)
    
    print('Fitting\n-------')
    for i in range(args.numsteps):
        value, opt_state = update(i, opt_state)
        if args.verbose:
            print(f'loss = {value:{fmt}}') 
    print(f'mean squared error = {value:{fmt}}\n')

    a_fit, b_fit = params_fit = get_params(opt_state)
    print('Fit parameters\n--------------')
    print(f'a_fit  =\n{a_fit},\nb_fit  = {b_fit:{fmt}}')   

if __name__ == '__main__':
    main()
