import numpy as np
import matplotlib.pyplot as plt

from asterion.models import GlitchModel, HeGlitchFunction, CZGlitchFunction, \
    AsyFunction, distribution
from asterion.inference import Inference

SEED = 0

_delta_nu = ()  # Placeholder for delta_nu
_nu_max = ()  # Placeholder for nu_max

# Define the models for the background, He and CZ glitches
background = AsyFunction(_delta_nu)
he_glitch = HeGlitchFunction(_nu_max)
cz_glitch = CZGlitchFunction(_nu_max)

# Define the glitch model
model = GlitchModel(background, he_glitch, cz_glitch)

# Define the inference class
infer = Inference(model, seed=SEED)

def fit_glitch(n, nu, delta_nu, nu_max, num_samples=1000, num_chains=5):
    """Fits the glitch.

    Args:
        n (array_like of int): Radial order.
        nu (array_like): Mode frequency for each radial order.
        delta_nu (tuple): Prior for delta_nu (<mean>, <standard deviation>).
        nu_max (tuple): Prior for nu_max (<mean>, <standard deviation>).
        num_samples (int, optional): Number of samples per chain. Defaults to 1000.
        num_chains (int, optional): Number of chains. Defaults to 5.

    Returns:
        asterior.inference.Results: Results object.
    """
    # Resets the priors - converts to distributions
    background.delta_nu = distribution(delta_nu)
    he_glitch.nu_max = distribution(nu_max)
    cz_glitch.nu_max = distribution(nu_max)
    
    # Samples with new data
    infer.sample(num_samples=num_samples, num_chains=num_chains,
                 model_args=(n,), model_kwargs={'nu_obs': nu})
    
    infer.posterior_predictive(model_args=(n,), 
                               model_kwargs={'nu_obs': nu, 'pred': True})

    # Get results
    result = infer.get_results()
    return result

if __name__ == "__main__":
    # Load data
    data = np.loadtxt('data/three_stars.csv', delimiter=',')
    n = np.arange(10, 28)
    
    nu = data[:, 12:30]
    delta_nu_fit = data[:, 1]
    nu_max_sca = data[:, 2]
    surface_he = data[:, -1]
    
    results = []

    for i in range(data.shape[0]):
        # For each model we need a rough prior for delta_nu and nu_max
        delta_nu = (delta_nu_fit[i], 0.5)
        nu_max = (nu_max_sca[i], 0.05*nu_max_sca[i])
        result = fit_glitch(n, nu[i], delta_nu, nu_max)
        results.append(result)

    fig = plt.figure(figsize=(6.4, 9.6))
    
    # Plot He glitch
    ax = fig.add_subplot(2, 1, 1)
    
    for i, result in enumerate(results):
        # Use the model's glitch plotting function
        # ========================================
        # You could do this your own way instead, something like:
        # dnu_he = results.data.posterior['dnu_he'].median(dim=('chain', 'draw'))
        # ax.plot(n, dnu_he)
        ax = model.plot_glitch(result.data, kind='He', observed=False, ax=ax)
        # We need to get the last line drawn and set its color and label.
        # You don't need to do this if you plot it your own way.
        line = ax.get_lines()[-1]
        line.set_color(f'C{i}')
        line.set_label(f'surface He = {surface_he[i]:.2f}')
    ax.legend()
    
    # Plot CZ glitch
    ax = fig.add_subplot(2, 1, 2)

    for i, result in enumerate(results):
        # The same as above but for dnu_cz
        ax = model.plot_glitch(result.data, kind='CZ', observed=False, ax=ax)
        line = ax.get_lines()[-1]
        line.set_color(f'C{i}')
        line.set_label(f'surface He = {surface_he[i]:.2f}')
    ax.legend()
    
    plt.show()
