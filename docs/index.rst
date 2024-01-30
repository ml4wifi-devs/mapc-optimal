Optimal solution for IEEE 802.11 MAPC Coordinated Spatial Reuse (C-SR) problem
==============================================================================

``mapc-optimal`` is a tool for finding the optimal solution of the
Multi-Access Point Coordination (MAPC) scheduling problem with
coordinated spatial reuse (C-SR) for IEEE 802.11 networks. It provides a
linear programming solution to find the upper bound on network
performance. A detailed description can be found in:

-  TODO

Features
--------

-  **Calculation of optimal scheduling**: Calculate the best
   transmission configurations and the corresponding time division that
   enhance the network performance.
-  **Two optimization criteria**: Find the optimal solution for two
   optimization criteria: maximizing the sum of the throughput of all
   nodes in the network and maximizing the minimum throughput of all
   nodes in the network.
-  **Modulation and coding scheme (MCS) optimization**: Select the
   optimal MCS for each transmission.
-  **Transmission power selection**: Set the appropriate transmit power
   to maximize network performance.
-  **Versatile network configuration**: Define network settings by
   specifying network nodes, available MCSs, channel model parameters,
   and transmission power levels.


Installation
------------

The package can be installed using pip:

.. code:: bash

   pip install mapc-optimal

Usage
-----

The main functionality is provided by the :class:`mapc_optimal.Solver` class.
This class manages the process of finding
the optimal solution. Example usage:

.. code:: python

   from mapc_optimal import Solver

   # Define your network
   # ...

   solver = Solver(stations, access_points)
   configurations, rate = solver(path_loss)

where ``stations`` and ``access_points`` are lists of numbers
representing the stations and access points (APs) in the network,
respectively. Identifiers of the stations and APs should be unique and
cover the range from :math:`0` to :math:`n - 1` (where :math:`n` is the
total number of nodes in the network). The ``path_loss`` is an
:math:`n \times n` matrix representing the path loss between each pair
of nodes in the network. The solver returns calculated configurations
and the corresponding total throughput of the network.

The :class:`mapc_optimal.Solver`  class can be further configured by passing additional
arguments.


.. code:: python

   configurations, rate, objectives = solver(path_loss, return_objective=True)

For a more detailed example, refer to the test case in
``test/test_solver.py``.




How to reference ``mapc-optimal``?
----------------------------------

::

   TODO

.. toctree::
    :hidden:

    genindex

