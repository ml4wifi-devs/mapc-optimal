from itertools import product

import networkx as nx
from numpy.typing import NDArray
from mapc_sim.constants import *

from mapc_optimal.solver.master import Master
from mapc_optimal.solver.pricing import Pricing
from mapc_optimal.solver.utils import dbm_to_lin, lin_to_dbm


class Solver:
    def __init__(
            self,
            stations: list,
            access_points: list,
            mcs_values: int = len(DATA_RATES),
            mcs_data_rates: NDArray = DATA_RATES,
            min_snr: NDArray = MEAN_SNRS,
            default_tx_power: float = DEFAULT_TX_POWER,
            max_tx_power: float = MAX_TX_POWER,
            min_tx_power: float = MIN_TX_POWER,
            noise_floor: float = NOISE_FLOOR,
            min_throughput: float = 0.,
            opt_sum: bool = False,
            max_iterations: int = 10
    ) -> None:
        self.stations = stations
        self.access_points = access_points
        self.mcs_data_rates = mcs_data_rates
        self.max_iterations = max_iterations

        self.master = Master(
            min_throughput=min_throughput,
            opt_sum=opt_sum
        )
        self.pricing = Pricing(
            mcs_values=range(mcs_values),
            mcs_data_rates=mcs_data_rates,
            min_sinr=dbm_to_lin(min_snr),
            default_tx_power=dbm_to_lin(default_tx_power).item(),
            max_tx_power=dbm_to_lin(max_tx_power).item(),
            min_tx_power=dbm_to_lin(min_tx_power).item(),
            noise_floor=dbm_to_lin(noise_floor).item(),
            opt_sum=opt_sum
        )

    def _generate_data(self, path_loss: NDArray) -> dict:
        graph = nx.DiGraph()

        for s in self.stations:
            graph.add_node(f'STA_{s}', type='STA')

        for a in self.access_points:
            graph.add_node(f'AP_{a}', type='AP')

        for s in self.stations:
            best_pl = float('inf')
            best_ap = None

            for a in self.access_points:
                if path_loss[a, s] < best_pl:
                    best_pl = path_loss[a, s].item()
                    best_ap = a

            graph.add_edge(f'AP_{best_ap}', f'STA_{s}')

        return {
            'graph': graph,
            'stations': [v for v in graph.nodes if graph.nodes[v]['type'] == 'STA'],
            'access_points': [v for v in graph.nodes if graph.nodes[v]['type'] == 'AP'],
            'links': list(graph.edges),
            'link_node_a': {e: e[0] for e in graph.edges},  # APs
            'link_node_b': {e: e[1] for e in graph.edges},  # STAs
            'link_path_loss': {(f'AP_{a}', f'STA_{s}'): path_loss[a, s].item() for a, s in product(self.access_points, self.stations)}
        }

    def __call__(self, path_loss: NDArray) -> tuple:
        path_loss = dbm_to_lin(path_loss)
        problem_data = self._generate_data(path_loss)

        configuration = self.pricing.initial_configuration(
            stations=problem_data['stations'],
            link_node_a=problem_data['link_node_a'],
            link_path_loss=problem_data['link_path_loss'],
            graph=problem_data['graph']
        )

        iteration = 0
        master_goal_best = -float('inf')
        pricing_goal_best = -float('inf')

        while iteration < self.max_iterations:
            master_result, master_goal = self.master(
                stations=problem_data['stations'],
                link_node_b=problem_data['link_node_b'],
                conf_links=configuration['conf_links'],
                conf_link_rates=configuration['conf_link_rates'],
                conf_total_rates=configuration['conf_total_rates'],
                confs=configuration['confs']
            )

            configuration, pricing_goal = self.pricing(
                dual_alpha=master_result['alpha'],
                dual_beta=master_result['beta'],
                dual_gamma=master_result['gamma'],
                stations=problem_data['stations'],
                access_points=problem_data['access_points'],
                links=problem_data['links'],
                link_node_a=problem_data['link_node_a'],
                link_node_b=problem_data['link_node_b'],
                link_path_loss=problem_data['link_path_loss'],
                configuration=configuration
            )

            if master_goal > master_goal_best or pricing_goal > pricing_goal_best:
                master_goal_best = max(master_goal, master_goal_best)
                pricing_goal_best = max(pricing_goal, pricing_goal_best)
            else:
                break

            iteration += 1

        result = {
            'links': configuration['conf_links'],
            'link_rates': configuration['conf_link_rates'],
            'total_rates': configuration['conf_total_rates'],
            'tx_power': {c: {l: lin_to_dbm(p).item() for l, p in tx_power.items()} for c, tx_power in configuration['conf_link_tx_power'].items()},
            'shares': master_result['shares']
        }

        return result, sum(result['total_rates'][c] * result['shares'][c] for c in result['shares'])
