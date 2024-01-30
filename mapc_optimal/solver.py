from itertools import product

import numpy as np

from mapc_optimal.constants import DATA_RATES, MAX_TX_POWER, MIN_SNRS, MIN_TX_POWER, NOISE_FLOOR
from mapc_optimal.main import Main
from mapc_optimal.pricing import Pricing
from mapc_optimal.utils import dbm_to_lin, lin_to_dbm


class Solver:
    """The solver class coordinating the overall process of finding the optimal
    solution. It initializes the solver, sets up the network configuration,
    and manages the iterations.

    Examples
    --------

    .. code-block:: python

       from mapc_optimal import Solver

       # Define your network
       # ...

       solver = Solver(stations, access_points)
       configurations, rate = solver(path_loss)


    .. note::
        The solver requires the path loss between each pair of nodes in the
        network. The reason for this is that the solver should be independent
        of the channel model used. Therefore, the path loss must be
        calculated beforehand. Note that if you do not require a specific
        channel model, you can use the provided function to calculate the
        path loss using the TGax channel model based on the positions of the
        nodes:


        .. code:: python

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


    """
    def __init__(
            self,
            stations: list,
            access_points: list,
            mcs_values: int = len(DATA_RATES),
            mcs_data_rates: list = DATA_RATES,
            min_snr: list = MIN_SNRS,
            max_tx_power: float = MAX_TX_POWER,
            min_tx_power: float = MIN_TX_POWER,
            noise_floor: float = NOISE_FLOOR,
            min_throughput: float = 0.,
            opt_sum: bool = False,
            max_iterations: int = 100,
            epsilon: float = 1e-5
    ) -> None:
        """
        .. warning::
            Identifiers of the stations and APs should be unique and
            cover the range from :math:`0` to :math:`n - 1` (where :math:`n` is the
            total number of nodes in the network).

        Parameters
        ----------
        stations: list
            Lists of numbers representing the stations.
        access_points: list
            Lists of numbers representing the access points (APs) in the network.
        mcs_values: int
            A number of MCS values available in the network [*12 by default*].
        mcs_data_rates: list
            A list of data rates corresponding to the MCS values (Mb/s) [*IEEE 802.11ax single stream with
            20MHz bandwidth and 800 ns GI data rates by default*].
        min_snr: list
            The minimum SNR required for a successful transmission
            (dB) for each MCS value [*empirically determined in ns-3 simulations by default*]
        max_tx_power: float
            The maximum transmission power (dBm) available [*20 dBm by default*].
        min_tx_power: float
            The minimum transmission power (dBm) that can be used [*10 dBm by default*].
        noise_floor: float
            The level of noise in the environment (dBm) [*-93.97 dBm by default*].
        min_throughput: float
            The minimum throughput required for each node (Mb/s) while maximizing the total throughput
            [*0 Mb/s by default*].
        opt_sum: bool
            A boolean value indicating whether to maximize the sum of the throughput of all nodes in the network
            (`True`) or the minimum throughput of all nodes in the network (`False`) [*`False` by default*].
        max_iterations: int
            The maximum number of iterations of the solver [*100 by default*].
        epsilon: float
             The minimum value of the pricing objective function to continue the iterations [*1e-5 by default*].

        """
        self.stations = stations
        self.access_points = access_points
        self.mcs_values = range(mcs_values)
        self.mcs_data_rates = mcs_data_rates
        self.min_sinr = dbm_to_lin(min_snr)
        self.max_tx_power = dbm_to_lin(max_tx_power).item()
        self.min_tx_power = dbm_to_lin(min_tx_power).item()
        self.noise_floor = dbm_to_lin(noise_floor).item()
        self.min_throughput = min_throughput
        self.opt_sum = opt_sum
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        self.main = Main(
            min_throughput=self.min_throughput,
            opt_sum=self.opt_sum
        )
        self.pricing = Pricing(
            mcs_values=self.mcs_values,
            mcs_data_rates=self.mcs_data_rates,
            min_sinr=self.min_sinr,
            max_tx_power=self.max_tx_power,
            min_tx_power=self.min_tx_power,
            noise_floor=self.noise_floor,
            opt_sum=self.opt_sum
        )

    def _tx_possible(self, path_loss: float) -> bool:
        return self.max_tx_power >= self.min_sinr[0] * path_loss * self.noise_floor

    def _generate_data(self, path_loss: np.ndarray) -> dict:
        links = []

        for s in self.stations:
            best_pl = float('inf')
            best_ap = None

            for a in self.access_points:
                if path_loss[a, s] < best_pl:
                    best_pl = path_loss[a, s]
                    best_ap = a

            if self._tx_possible(best_pl):
                links.append((f'AP_{best_ap}', f'STA_{s}'))

        problem_data = {
            'stations': [f'STA_{s}' for s in self.stations],
            'access_points': [f'AP_{a}' for a in self.access_points],
            'links': links,
            'link_node_a': {l: l[0] for l in links},  # APs
            'link_node_b': {l: l[1] for l in links},  # STAs
        }

        link_path_loss = {(f'AP_{a}', f'STA_{s}'): path_loss[a, s].item() for a, s in product(self.access_points, self.stations)}
        max_interference = {}

        for l in links:
            a, s = problem_data['link_node_a'][l], problem_data['link_node_b'][l]

            for m in self.mcs_values:
                max_interference[l, m] = sum(
                    self.max_tx_power * (self.min_sinr[m] * link_path_loss[l] / link_path_loss[i, s]) +
                    self.min_sinr[m] * link_path_loss[l] * self.noise_floor
                    for i in problem_data['access_points'] if i != a
                )

        problem_data['link_path_loss'] = link_path_loss
        problem_data['max_interference'] = max_interference
        return problem_data

    def __call__(self, path_loss: np.ndarray, return_objectives: bool = False) -> tuple:
        """Run optimization.


        Parameters
        ----------
        path_loss
            TODO
        return_objectives
            TODO

        Returns
        -------
        tuple
            Solution. Additionally, the solver can return a list of the pricing objective
            values for each iteration. It can be useful to check if the solver has
            converged. To do so, set the ``return_objective`` argument to ``True``
            when calling the solver.

        """
        path_loss = dbm_to_lin(path_loss)
        problem_data = self._generate_data(path_loss)

        if len(problem_data['links']) == 0:
            if return_objectives:
                return {}, 0, []
            else:
                return {}, 0.

        configuration = self.pricing.initial_configuration(
            links=problem_data['links'],
            link_path_loss=problem_data['link_path_loss']
        )

        pricing_objectives = []

        for _ in range(self.max_iterations):
            main_result, main_objective = self.main(
                stations=problem_data['stations'],
                link_node_b=problem_data['link_node_b'],
                conf_links=configuration['conf_links'],
                conf_link_rates=configuration['conf_link_rates'],
                conf_total_rates=configuration['conf_total_rates'],
                confs=configuration['confs']
            )

            configuration, pricing_objective = self.pricing(
                dual_alpha=main_result['alpha'],
                dual_beta=main_result['beta'],
                dual_gamma=main_result['gamma'],
                stations=problem_data['stations'],
                access_points=problem_data['access_points'],
                links=problem_data['links'],
                link_node_a=problem_data['link_node_a'],
                link_node_b=problem_data['link_node_b'],
                link_path_loss=problem_data['link_path_loss'],
                max_interference=problem_data['max_interference'],
                configuration=configuration
            )
            pricing_objectives.append(pricing_objective)

            if pricing_objective <= self.epsilon:
                break

        result = {
            'links': configuration['conf_links'],
            'link_rates': configuration['conf_link_rates'],
            'total_rates': configuration['conf_total_rates'],
            'tx_power': {c: {l: lin_to_dbm(p).item() for l, p in tx_power.items()} for c, tx_power in configuration['conf_link_tx_power'].items()},
            'shares': main_result['shares']
        }
        total_rate = sum(result['total_rates'][c] * result['shares'][c] for c in result['shares'])

        if return_objectives:
            return result, total_rate, pricing_objectives
        else:
            return result, total_rate
