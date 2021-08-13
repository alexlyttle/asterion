.. asterion documentation master file, created by
   sphinx-quickstart on Thu Aug  5 16:26:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Asterion documentation
=====================================

A Bayesian package for modelling asteroseismic oscillation mode frequencies.

Introduction
------------

Do you have some asteroseismic mode frequencies for a particular star? Do you want to fit a model to those modes?
You need not look any further because this is the Python package for you. With this package you can fit a model
of the glitch in mode frequencies caused by the helium-II ionisation zone and base of the convective zone in the
stellar envelope. This model yields posterior estimates for the amplitudes of these glitches, which you can 
use to study surface helium content [references].

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :name: guidetoc

   guide/installation
   guide/getting_started
   guide/api

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials
   :name: tutorialstoc

   tutorials/*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Development
   :name: devtoc

   dev/*
   GitHub repository <https://github.com/alexlyttle/asterion>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
