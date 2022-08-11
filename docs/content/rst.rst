Technical Documentation
=======================

For the documentation of specific methods consider the docstrings and inline comments of these methods.

Main
-----------------------
The main method is structured as follows:

   1. Argument Parsing
   2. Data Loading
   3. Model Preparation and loading
   4. Optimizer and Loss Setup
   5. CUDA Setup
   6. Train or Predict


LAMP
--------------------

The Model can be found in the ``src/lamp/Models.py``

.. note:: In the LAMP directory we only contributed the RESNETs in the ``Decoders.py`` and ``Models.py``



Evaluation
----------------------
To reconstruct our evaluation process, use the ``plots/plot_training.ipynb`` file. The averaged results are attached in the results directory as CSV files.

