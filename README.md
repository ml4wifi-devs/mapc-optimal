# Optimal solution for IEEE 802.11 MAPC Coordinated Spatial Reuse (C-SR) problem

`mapc-optimal` is a tool for finding the optimal solution of the Multi-Access Point Coordination (MAPC) scheduling 
problem with coordinated spatial reuse (C-SR) for IEEE 802.11 networks. It provides a mixed-integer linear programming
(MILP) solution to find the upper bound on network performance. A detailed description can be found in:

- TODO

## Features

- **Calculation of optimal scheduling**: Calculate the best transmission configurations and the corresponding time 
  division that enhance the network performance.
- **Two optimization criteria**: Find the optimal solution for two optimization criteria: maximizing the sum of the 
  throughput of all nodes in the network and maximizing the minimum throughput of all nodes in the network.
- **Modulation and coding scheme (MCS) selection**: Select the optimal MCS for each transmission.
- **Transmission power selection**: Set the appropriate transmit power to maximize network performance.
- **Versatile network configuration**: Define network settings by specifying network nodes, available MCSs, 
  and transmission power levels.

## Installation

The package can be installed using pip:

```bash
pip install mapc-optimal
```

## Usage

The main functionality is provided by the `Solver` class in `mapc_optimal/solver.py`. This class manages the process of 
finding the optimal solution. Example usage:

```python
from mapc_optimal import Solver

# Define your network
# ...

solver = Solver(stations, access_points)
configurations, rate = solver(path_loss)
```

where `stations` and `access_points` are lists of numbers representing the stations and access points (APs) in the 
network, respectively. Identifiers of the stations and APs should be unique and cover the range from $0$ to $n - 1$
(where $n$ is the total number of nodes in the network). The `path_loss` is an $n \times n$ matrix representing the 
path loss between each pair of nodes in the network. The solver returns calculated configurations and the corresponding
total throughput of the network. The `Solver` class can be further configured by passing additional arguments to the 
constructor. Full list of arguments can be found in the documentation. 

Additionally, the solver can return a list of the pricing objective values for each iteration. It can be useful to 
check if the solver has converged. To do so, set the `return_objective` argument to `True` when calling the solver.

```python
configurations, rate, objectives = solver(path_loss, return_objective=True)
```

For a more detailed example, refer to the test case in `test/test_solver.py`.

**Note:** The underlying MILP solver can significantly affect the performance of the tool. By default, the solver 
uses the `CBC` solver from the `PuLP` package. However, we recommend using a better solver, such as `CPLEX`.

## Repository Structure

The repository is structured as follows:

- `mapc_optimal/`: The main package of the tool.
  - `constants.py`: Default values of the parameters used in the solver.
  - `main.py`: The formulation of the main problem solving the selection and division of configurations.
  - `pricing.py`: The pricing algorithm used to propose new configurations for the main problem.
  - `solver.py`: The solver class coordinating the overall process of finding the optimal solution. It initializes the 
     solver, sets up the network configuration, and manages the iterations.
  - `utils.py`: Utility functions, including the function for calculation of the path loss from node positions using 
    the TGax channel model.
- `test/`: Unit tests with example usage of the tool.

## How to reference `mapc-optimal`?

```
TODO
```
