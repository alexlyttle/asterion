.. ./README.rst file, created by
   scripts/make_readme.py on Mon Aug 23 14:35:07 2021 UTC.
   ================ DO NOT MODIFY THIS FILE! =================
   It is generated automatically as a part of a GitHub Action.
   Any changes should be made to
   ./README.rst.src instead.
   ===========================================================

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

Installation
============

This module is actively being developed. To install the latest build, run the following command,

.. code-block:: shell

    pip install git+https://github.com/alexlyttle/asterion@master#egg=asterion

An official release on PyPI for this package will be coming soon.


Getting started
===============

[text]


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

