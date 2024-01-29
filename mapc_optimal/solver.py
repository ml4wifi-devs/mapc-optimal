from itertools import product

import networkx as nx
import numpy as np

from mapc_optimal.constants import DATA_RATES, MAX_TX_POWER, MIN_SNRS, MIN_TX_POWER, NOISE_FLOOR
from mapc_optimal.master import Master
from mapc_optimal.pricing import Pricing
from mapc_optimal.utils import dbm_to_lin, lin_to_dbm


class Solver:
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
        self.stations = stations
        self.access_points = access_points
        self.mcs_values = range(mcs_values)
        self.mcs_data_rates = np.array(mcs_data_rates).astype(float)
        self.min_sinr = dbm_to_lin(min_snr)
        self.max_tx_power = dbm_to_lin(max_tx_power).item()
        self.min_tx_power = dbm_to_lin(min_tx_power).item()
        self.noise_floor = dbm_to_lin(noise_floor).item()
        self.min_throughput = min_throughput
        self.opt_sum = opt_sum
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        self.master = Master(
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
        graph = nx.DiGraph()

        for s in self.stations:
            best_pl = float('inf')
            best_ap = None

            for a in self.access_points:
                if path_loss[a, s] < best_pl:
                    best_pl = path_loss[a, s].item()
                    best_ap = a

            if self._tx_possible(best_pl):
                graph.add_edge(f'AP_{best_ap}', f'STA_{s}')

        problem_data = {
            'graph': graph,
            'stations': [v for v in graph.nodes if 'STA' in v],
            'access_points': [v for v in graph.nodes if 'AP' in v],
            'links': list(graph.edges),
            'link_node_a': {e: e[0] for e in graph.edges},  # APs
            'link_node_b': {e: e[1] for e in graph.edges},  # STAs
        }

        link_path_loss = {(f'AP_{a}', f'STA_{s}'): path_loss[a, s].item() for a, s in product(self.access_points, self.stations)}
        max_interference = {}

        for l in graph.edges:
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
        path_loss = dbm_to_lin(path_loss)
        problem_data = self._generate_data(path_loss)

        if len(problem_data['links']) == 0:
            return {}, 0.

        configuration = self.pricing.initial_configuration(
            stations=problem_data['stations'],
            link_path_loss=problem_data['link_path_loss'],
            graph=problem_data['graph']
        )

        pricing_objectives = []

        for _ in range(self.max_iterations):
            master_result, master_objective = self.master(
                stations=problem_data['stations'],
                link_node_b=problem_data['link_node_b'],
                conf_links=configuration['conf_links'],
                conf_link_rates=configuration['conf_link_rates'],
                conf_total_rates=configuration['conf_total_rates'],
                confs=configuration['confs']
            )

            configuration, pricing_objective = self.pricing(
                dual_alpha=master_result['alpha'],
                dual_beta=master_result['beta'],
                dual_gamma=master_result['gamma'],
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
            'shares': master_result['shares']
        }
        total_rate = sum(result['total_rates'][c] * result['shares'][c] for c in result['shares'])

        if return_objectives:
            return result, total_rate, pricing_objectives
        else:
            return result, total_rate
