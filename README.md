# Optimal solution for IEEE 802.11 MAPC Coordinated Spatial Reuse (C-SR) problem

`mapc-optimal` is a tool for finding the optimal solution of the Multi-Access Point Coordination (MAPC) scheduling 
problem with coordinated spatial reuse (C-SR) for IEEE 802.11 networks. It provides a mixed-integer linear programming 
solution to find the upper bound on network performance. A detailed description can be found in:

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

## Repository Structure

The repository is structured as follows:

- `mapc_optimal/`: The main package of the tool.
  - `constants.py`: Default values of the parameters used in the solver.
  - `master.py`: The formulation of the master problem solving the selection and division of configurations.
  - `pricing.py`: The pricing algorithm used to propose new configurations for the master problem.
  - `solver.py`: The solver class coordinating the overall process of finding the optimal solution. It initializes the 
     solver, sets up the network configuration, and manages the iterations.
  - `utils.py`: Utility functions, including the function for calculation of the path loss from node positions using 
    the TGax channel model.
- `test/`: Unit tests with example usage of the tool.

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
total throughput of the network.

### Additional Configuration

The `Solver` class can be further configured by passing additional arguments:

- `mcs_values`: A number of MCS values available in the network [*12 by default*].
- `mcs_data_rates`: A list of data rates corresponding to the MCS values (Mb/s) [*IEEE 802.11ax single stream with 20 
  MHz bandwidth and 800 ns GI data rates by default*].
- `min_snr`: The minimum SNR required for a successful transmission (dB) for each MCS value [*empirically determined 
  in ns-3 simulations by default*].
- `max_tx_power`: The maximum transmission power (dBm) available [*20 dBm by default*].
- `min_tx_power`: The minimum transmission power (dBm) that can be used [*10 dBm by default*].
- `noise_floor`: The level of noise in the environment (dBm) [*-93.97 dBm by default*].
- `min_throughput`: The minimum throughput required for each node (Mb/s) while maximizing the total throughput 
  [*0 Mb/s by default*].
- `opt_sum`: A boolean value indicating whether to maximize the sum of the throughput of all nodes in the network 
  (`True`) or the minimum throughput of all nodes in the network (`False`) [*`False` by default*].
- `max_iterations`: The maximum number of iterations of the solver [*100 by default*].
- `epsilon`: The minimum value of the pricing objective function to continue the iterations [*1e-5 by default*].

Additionally, the solver can return a list of the pricing objective values for each iteration. It can be useful to 
check if the solver has converged. To do so, set the `return_objective` argument to `True` when calling the solver.

```python
configurations, rate, objectives = solver(path_loss, return_objective=True)
```

For a more detailed example, refer to the test case in `test/test_solver.py`.

### Path Loss Calculation

The solver requires the path loss between each pair of nodes in the network. The reason for this is that the solver 
should be independent of the channel model used. Therefore, the path loss must be calculated beforehand. Note that 
if you do not require a specific channel model, you can use the provided function to calculate the path loss 
using the TGax channel model based on the positions of the nodes:
     
```python
import numpy as np    
from mapc_optimal import position_to_path_loss

# Positions of the nodes as an array of `x` and `y` coordinates. `i`-th row represents the position 
# of the node with identifier `i` in the `stations` and `access_points` lists.
pos = np.array([
    [x_0, y_0],
    [x_1, y_1],
    ...
    [x_n-1, y_n-1]
])

# A matrix representing the walls in the environment (1 - wall, 0 - no wall between nodes `i` and `j`).
walls = np.zeros((n, n))
walls[i_0, j_0] = 1
walls[i_1, j_1] = 1
...
walls[i_m, j_m] = 1

# n x n matrix representing the path loss between each pair of nodes.
path_loss = position_to_path_loss(pos, walls)
```

## How to reference `mapc-optimal`?

```
TODO
```
