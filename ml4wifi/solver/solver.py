import networkx as nx
import numpy as np

from ml4wifi.solver.master import Master
from ml4wifi.solver.pricing import Pricing
from ml4wifi.solver.utils import dbm_to_lin, lin_to_dbm
from ml4wifi.wifi import *


class Solver:
    def __init__(
            self,
            stations: list,
            access_points: list,
            mcs_values: int = len(DATA_RATES),
            mcs_data_rates: np.ndarray = DATA_RATES,
            min_snr: np.ndarray = MEAN_SNRS,
            max_tx_power: float = MAX_TX_POWER,
            noise_floor: float = NOISE_FLOOR,
            min_throughput: float = 0.,
            opt_sum: bool = False,
            max_iterations: int = 10
    ) -> None:
        self.stations = stations
        self.access_points = access_points
        self.mcs_data_rates = mcs_data_rates
        self.min_snr_dbm = min_snr
        self.max_iterations = max_iterations

        self.master = Master(
            min_throughput=min_throughput,
            opt_sum=opt_sum
        )
        self.pricing = Pricing(
            mcs_values=list(range(mcs_values)),
            mcs_data_rates=mcs_data_rates,
            min_snr=dbm_to_lin(min_snr),
            max_tx_power=dbm_to_lin(max_tx_power).item(),
            noise_floor=dbm_to_lin(noise_floor).item(),
            opt_sum=opt_sum
        )

    def _best_thr(self, path_loss: float) -> float:
        path_loss = lin_to_dbm(path_loss)
        snr = DEFAULT_TX_POWER - path_loss - NOISE_FLOOR
        return self.mcs_data_rates[(snr >= self.min_snr_dbm).sum()]

    def _initial_compatible_sets(
            self,
            stations: list,
            links: list,
            link_node_b: dict,
            link_path_loss: dict,
            graph: nx.DiGraph
    ) -> dict:
        configuration = {}

        # Compatible sets / configurations:
        # to be extended with iterations of the algorithm
        # we start with very simple compatible sets:
        # -> each set is related to transmission to a single STA from its AP
        configuration['conf_num'] = len(stations) + 1
        
        # we number the configurations starting from 1
        configuration['confs'] = range(1, len(stations) + 1)

        # links used in compatible sets
        configuration['conf_links'] = {c: None for c in configuration['confs']}

        for i, s in enumerate(stations, 1):
            sta_link = list(graph.in_edges(s))
            configuration['conf_links'][i] = [sta_link[0]]

        # link rates for compatible sets
        configuration['conf_link_rates'] = {
            (c, l): self._best_thr(link_path_loss[l]) for c in configuration['confs'] for l in links
        }

        # station rates for compatible sets
        configuration['conf_sta_rates'] = {
            (c, s): sum(configuration['conf_link_rates'][c, l] for l in configuration['conf_links'][c] if link_node_b[l] == s)
            for c in configuration['confs']
            for s in stations
        }

        # link tx power for compatible sets
        configuration['conf_link_tx_power'] = {
            (c, l): dbm_to_lin(DEFAULT_TX_POWER) for c in configuration['confs'] for l in links
        }

        # total rates for compatible sets
        configuration['conf_total_rates'] = {
            c: sum(configuration['conf_sta_rates'][c, s] for s in stations) for c in configuration['confs']
        }
        
        return configuration

    def _generate_data(self, path_loss: np.ndarray) -> dict:
        graph = nx.DiGraph()

        for sta in self.stations:
            graph.add_node(f'STA_{sta}', type='STA')

        for ap in self.access_points:
            graph.add_node(f'AP_{ap}', type='AP')

        for sta in self.stations:
            best_pl = np.inf
            best_ap = None

            for ap in self.access_points:
                if path_loss[ap, sta] < best_pl:
                    best_pl = path_loss[ap, sta].item()
                    best_ap = ap

            graph.add_edge(f'AP_{best_ap}', f'STA_{sta}', path_loss=best_pl)

        return {
            'graph': graph,
            'stations': [v for v in graph.nodes if graph.nodes[v]['type'] == 'STA'],
            'access_points': [v for v in graph.nodes if graph.nodes[v]['type'] == 'AP'],
            'links': list(graph.edges),
            'link_node_a': {e: e[0] for e in graph.edges},  # APs
            'link_node_b': {e: e[1] for e in graph.edges},  # STAs
            'link_path_loss': {e: graph.edges[e]['path_loss'] for e in graph.edges},
        }

    def __call__(self, path_loss: np.ndarray) -> dict:
        path_loss = dbm_to_lin(path_loss)
        problem_data = self._generate_data(path_loss)

        configuration = self._initial_compatible_sets(
            stations=problem_data['stations'],
            links=problem_data['links'],
            link_node_b=problem_data['link_node_b'],
            link_path_loss=problem_data['link_path_loss'],
            graph=problem_data['graph']
        )

        iteration = 0
        master_goal_best = -np.inf
        pricing_goal_best = -np.inf

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

        return {
            'links': configuration['conf_links'],
            'link_rates': configuration['conf_link_rates'],
            'link_tx_power': configuration['conf_link_tx_power'],
            'shares': master_result['shares']
        }
