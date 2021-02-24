"""
Basic linear model.
"""
import jax
import jax.numpy as jnp

from regression import init_optimizer, loss_fn, make_targets, get_update_fn, \
    make_plot
from parser import parse_args

def model(params, inputs):
    return params[0] + jnp.dot(params[1], inputs)

def main():
    args = parse_args(__doc__)
    fmt = args.format

    a_true, b_true = params_true = (-5., 2.)
    print('True parameters\n---------------')
    print(f'a_true = {a_true:{fmt}}, b_true = {b_true:{fmt}}\n')

    n_data = 100
    x = jnp.linspace(0, 1, n_data)
    y_obs = make_targets(jax.random.PRNGKey(42),
                         params_true, x, model, scale=args.error)

    a_init, b_init = params_init = (-3., 1.)
    print('Initial parameters\n------------------')
    print(f'a_init = {a_init:{fmt}}, b_init = {b_init:{fmt}}\n')   

    opt_state, opt_update, get_params = init_optimizer(params_init, args.lrate)
    update = get_update_fn(opt_update, get_params, x, y_obs, model)
    
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

        y_true = model(params_true, x)
        y_fit = model(params_fit, x)
        
        fig, ax = plt.subplots()
        ax = make_plot(ax, x, y_obs, y_true, y_fit)

        plt.show()


if __name__ == '__main__':
    main()
