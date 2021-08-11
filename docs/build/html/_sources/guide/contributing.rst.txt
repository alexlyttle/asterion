Contributing
============

[text]

Tests
-----

[text]

Documentation
-------------

To modify and update the documentation you need to install the following module:

.. code-block:: shell

    pip install sphinx

Once you have made changes to docstrings, run the following commands to update the HTML documentation:

.. code-block:: shell

    cd docs
    make clean
    make html

**Optional**: If you have added a submodule or subpackage to ``asterion``, run the following command in the main project directory to update the API documentation:

.. code-block:: shell

    sphinx-apidoc -f -M -H "API reference" --tocfile api -t docs/source/_templates -o docs/source/guide asterion

This recursively searches ``asterion`` and generates a subsection for each submodule and subpackage. Then, update the HTML documentation.
