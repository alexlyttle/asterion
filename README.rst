.. image:: https://readthedocs.org/projects/asterion/badge/?version=latest&style=flat
    :target: https://asterion.readthedocs.io
    :alt: Documentation status
.. image:: https://github.com/alexlyttle/asterion/actions/workflows/main.yml/badge.svg
    :target: https://github.com/alexlyttle/asterion/actions/workflows/main.yml
    :alt: Build status
.. image:: https://img.shields.io/github/issues-closed/alexlyttle/asterion.svg
    :target: https://github.com/alexlyttle/asterion/issues
    :alt: Issues closed
.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/alexlyttle/asterion/blob/main/LICENSE
    :alt: License

########
Asterion
########

Warning: This package is an early build, use with caution!

Fit the asteroseismic helium-II ionisation zone glitch present in the mode frequencies of solar-like oscillators.

.. installation_label
Installation
============

This module is actively being developed. To install the latest build, run the following command,

.. code-block:: shell

    pip install git+https://github.com/alexlyttle/asterion@main#egg=asterion

An official release on PyPI for this package will be coming soon.

.. getting_started_label
Getting started
===============

Asterion is built to fit acoustic glitches in the asterosesimic mode frequencies of solar-like oscillators. Before you start, make sure you have all of the required inputs and any of the optional inputs.

Required Inputs
---------------

* Radial (l=0) mode frequencies, :math:`\nu`
* Frequency of maximum power, :math:`\nu_\max`, and its uncertainty
* Large frequency separation, :math:`\Delta\nu`, and its uncertainty

Optional Inputs
---------------

* Uncertainty on the mode frequencies, :math:`\sigma_\nu`
* Effective temperature of the star, :math:`T_\mathrm{eff}`
* Asymptotic frequency offset/phase, :math:`\epsilon`, and its uncertainty

Example
-------

Firstly, define your inputs, for example:

.. code-block:: python

    # Location and scale of a normal distribution
    nu_max = (2357.69, 25.0)
    delta_nu = (111.84, 0.1)
    teff = (5000.0, 200.0)

    n = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    nu = [1601.25, 1712.38, 1822.87, 1932.24,
          2042.3 , 2153.48, 2265.2 , 2377.14,
          2488.87, 2601.02, 2713.51, 2826.4 ,
          2939.56, 3052.67]
    nu_err = 0.01  # Can be a scalar or a value for each nu

Then, import Asterion and create the model.

.. code-block:: python

    from asterion import GlitchModel

    model = GlitchModel(n, nu_max, delta_nu, teff=teff)

Start inference. It is good practice to inspect the prior predictive check that it is sensible.

.. code-block:: python

    from asterion import Inference

    infer = Inference(model, nu, nu_err=nu_err, seed=10)
    infer.prior_predictive()  # <-- check prior is sensible
    prior_data = infer.get_data()
    ...  # <-- inspect prior data with e.g. asterion.plot_glitch (see below)

Once you are happy with the prior, sample from the posterior and inspect the posterior predictive.

.. code-block:: python

    # Sample from the posterior
    infer.sample()
    infer.posterior_predictive()

    # Save inference data
    data = infer.get_data()
    data.to_netcdf('results.nc')  # save inference data as NETCDF

You can use Asterion to make plots with the data and summarise in your favourite format (so long as it's either Pandas or Astropy). You can load the data and make plots and summaries any time using Arviz.

.. code-block:: python

    import asterion as ast
    import arviz as az  # <-- for loading the inference data
    import matplotlib.pyplot as plt

    data = az.from_netcdf('results.nc')  # <-- if loading the data elsewhere

    # Make plots to check posterior is sensible
    ast.plot_glitch(data, kind='He')
    ast.plot_glitch(data, kind='CZ')
    # E.g. a corner plot of the helium glitch parameters
    ast.plot_corner(data, var_names=['log_a_he', 'log_b_he', 'log_tau_he', 'phi_he'])

    # Save summary of results
    # Here all 0-dimensional parameters are saved in Astropy's
    # ECSV format which preserved data types and units
    table = ast.get_table(data, dims=(), fmt='astropy')
    table.write('data/summary.ecsv', overwrite=True)

    plt.show()

Check out the tutorials for more in-depth examples.

Notes
-----

* Variable names with the prefix :code:`'log_'` are base-10 logarithmic
* The :code:`seed` argument in :code:`GlitchModel` is used to sample from the prior on :math:`\tau` and should not affect inference.
* The :code:`seed` argument in :code:`Inference` is used for reproducibility and should not affect inference, but it is recommend you confirm this for yourself.

.. contributing_label
Contributing
============

If you find an issue with this package, please `search for or raise it on GitHub <https://github.com/alexlyttle/asterion/issues>`_.
If you would like to contribute to the package, please find an issue and let us know in the comments.

#. To start making changes, fork the repository using the link in the top-right of our `GitHub page <https://github.com/alexlyttle/asterion>`_.

#. Then, clone your fork,

   .. code-block:: shell

       git clone https://github.com/<your-username>/asterion.git
       cd asterion

#. We recommend setting up a virtual python environment to use while developing ``asterion``. For example,

   .. code-block:: shell

       mkdir ~/.venvs
       python -m venv ~/.venvs/asterion

   To activate the environment and work on the package,

   .. code-block:: shell

       source ~/.venvs/asterion/bin/activate

   When you have finished working, deactivate the environment with the command ``deactivate``.

#. Install the package (activate the virtual environment first if applicable),

   .. code-block:: shell

       pip install -e .

#. Add the main repository to your git environment,

   .. code-block:: shell

       git remote add upstream https://github.com/alexlyttle/asterion.git
       git remote -v

   The output should look like this,

   .. code-block::

       origin      https://github.com/<your-username>/asterion.git (fetch)
       origin      https://github.com/<your-username>/asterion.git (push)
       upstream    https://github.com/alexlyttle/asterion.git (fetch)
       upstream    https://github.com/alexlyttle/asterion.git (push)

#. Create a branch to work on your changes. Pull the latest version of the source code,

   .. code-block:: shell

       git checkout main
       git pull upstream main
   
   Then, create your branch,

   .. code-block:: shell

       git checkout -b <branch-name> 

#. Before adding your changes, run the unit tests (coming soon)...

#. Add and commit your changes. Please be specific in the commit message,

   .. code-block:: shell

       git add <added-or-modified-file>
       git commit -m "<description of your changes>"

#. Push changes to GitHub and open a pull request (you may open it as a draft if you are not ready for review),

   .. code-block:: shell

       git push origin <branch-name>
   
   Then, go `here <https://github.com/alexlyttle/asterion>`_ and click on the button "Compare and open a pull request" to submit your changes.

Tests
-----

Unit tests are coming soon.

Documentation
-------------

To modify and update the documentation you need to install the package with the ``docs`` option:

.. code-block:: shell

    pip install -e '.[docs]'

Once you have made changes to documentation, run the following commands to update the HTML documentation and check that the docs compile locally:

.. code-block:: shell

    cd docs
    make clean
    make html

**Optional**: If you have added a submodule or subpackage to ``asterion``, run the following command in the main project directory to update the API documentation:

.. code-block:: shell

    sphinx-apidoc -f -M -H "API reference" --tocfile api -t docs/source/_templates -o docs/source/guide asterion

This recursively searches ``asterion`` and generates a subsection for each submodule and subpackage. Then, build the docs to check it compiles locally.

