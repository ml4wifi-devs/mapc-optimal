Optimal solution for IEEE 802.11 MAPC Coordinated Spatial Reuse (C-SR) problem
==============================================================================

``mapc-optimal`` is a tool for finding the optimal solution of the
Multi-Access Point Coordination (MAPC) scheduling problem with
coordinated spatial reuse (C-SR) for IEEE 802.11 networks. It provides a
mixed-integer linear programming (MILP) solution to find the upper bound
on network performance. A detailed description can be found in:

-  TODO


Features
--------

-  **Calculation of optimal scheduling**: Calculate the best
   transmission configurations and the corresponding time division that
   enhance the network performance.
-  **Multiple optimization criteria**: Find the optimal solution for
   different optimization criteria: maximizing the sum of the throughput
   of all nodes in the network, maximizing the minimum throughput of all
   nodes in the network (optionally above a given baseline), maximizing
   the proportional fairness, and maximizing the vector of the node
   throughputs lexicographically.
-  **Modulation and coding scheme (MCS) selection**: Select the
   optimal MCS for each transmission.
-  **Transmission power selection**: Set the appropriate transmission power
   to maximize network performance.
-  **Versatile network configuration**: Define network settings by
   specifying network nodes, available MCSs, and transmission power levels.


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
respectively. The ``path_loss`` is an :math:`n \times n` matrix
representing the path loss between each pair of nodes in the network.
The solver returns calculated configurations and the total throughput
of the network. The :class:`mapc_optimal.Solver` class can be further
configured by passing additional arguments to the constructor.
The full list of arguments can be found in the documentation.

The optimization criterion is selected with the ``opt_type`` argument. Besides maximizing
the total throughput (``OptimizationType.SUM``) and the worst station throughput
(``OptimizationType.MAX_MIN``), the solver can maximize the vector of the station
throughputs lexicographically (``OptimizationType.LEXICOGRAPHIC``), i.e., once the worst
stations cannot be improved any further, it keeps improving the subsequent worst ones:

.. code:: python

   from mapc_optimal import OptimizationType, Solver

   solver = Solver(stations, access_points, opt_type=OptimizationType.LEXICOGRAPHIC)
   configurations, rate = solver(path_loss)

**Note** The lexicographic optimization solves a separate problem for each station in every
step of its outer loop, so it is significantly slower than the other criteria.

The solver can also guarantee that no station gets less than in some reference solution, e.g.,
the one obtained with a simulator or a learning agent. With ``OptimizationType.MAX_MIN_BASELINE``,
the worst station throughput is maximized while each station is required to reach its ``baseline``
rate. Since the rates of the reference solution can be hard to reproduce with configurations
generated from scratch, the configurations used by the reference solution can be added to the
initial ones with the ``initial_configurations`` argument. Each of them is a dictionary mapping
the (AP, station) pairs to the transmission power (dBm) used by the AP:

.. code:: python

   from mapc_optimal import OptimizationType, Solver

   solver = Solver(stations, access_points, opt_type=OptimizationType.MAX_MIN_BASELINE)
   configurations, rate = solver(
       path_loss,
       baseline={'STA_1': 10., 'STA_2': 12.},
       initial_configurations=[{(0, 1): 20., (1, 2): 16.}]
   )

If the baseline rates cannot be reached, the solver raises an exception. The baseline can also be
passed to the lexicographic optimization, where it sets the initial minimum throughput of each station.

Additionally, the solver can return a list of the pricing objective values for each
iteration. It can be useful to check if the solver has converged. To do so, set the
``return_objectives`` argument to ``True`` when calling the solver.

.. code:: python

   configurations, rate, objectives = solver(path_loss, return_objectives=True)

For a more detailed example, refer to the test case in ``test/test_solver.py``.

**Note** The underlying MILP solver can significantly affect the performance of the
tool. By default, the solver uses the ``CBC`` solver from the ``PuLP`` package.
However, we recommend using a better solver, such as ``CPLEX``.


How to reference ``mapc-optimal``?
----------------------------------

::

   TODO

.. toctree::
    :hidden:

    genindex

