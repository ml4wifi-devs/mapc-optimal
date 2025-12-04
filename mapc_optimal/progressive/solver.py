from collections import defaultdict
from itertools import product

import pulp as plp
from numpy.typing import NDArray

from mapc_optimal.constants import DATA_RATES, MAX_TX_POWER, MIN_SNRS, MIN_TX_POWER, NOISE_FLOOR
from mapc_optimal.progressive.main import Main
from mapc_optimal.progressive.pricing import Pricing
from mapc_optimal.utils import dbm_to_lin, lin_to_dbm


class Solver:
    def __init__(
            self,
            stations: list,
            access_points: list,
            channel_width: int = 20,
            mcs_data_rates: NDArray = None,
            min_snr: NDArray = None,
            max_tx_power: float = MAX_TX_POWER,
            min_tx_power: float = MIN_TX_POWER,
            noise_floor: float = NOISE_FLOOR,
            max_iterations: int = 100,
            epsilon: float = 1e-5,
            solver: plp.LpSolver = None
    ) -> None:
        if mcs_data_rates is None:
            mcs_data_rates = DATA_RATES[channel_width]

        if min_snr is None:
            min_snr = MIN_SNRS[channel_width]

        self.stations = stations
        self.access_points = access_points
        self.mcs_values = range(len(mcs_data_rates))
        self.mcs_data_rates = mcs_data_rates
        self.min_sinr = dbm_to_lin(min_snr)
        self.max_tx_power = dbm_to_lin(max_tx_power).item()
        self.min_tx_power = dbm_to_lin(min_tx_power).item()
        self.noise_floor = dbm_to_lin(noise_floor).item()
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.solver = solver or plp.PULP_CBC_CMD(msg=False)
        self.M = len(stations) * mcs_data_rates[-1]  # Maximum achievable throughput

        self.main = Main(solver=self.solver, M=self.M)
        self.pricing = Pricing(
            mcs_values=self.mcs_values,
            mcs_data_rates=self.mcs_data_rates,
            min_sinr=self.min_sinr,
            max_tx_power=self.max_tx_power,
            min_tx_power=self.min_tx_power,
            noise_floor=self.noise_floor,
            solver=self.solver
        )

    def _tx_possible(self, path_loss: float) -> bool:
        return self.max_tx_power >= self.min_sinr[0] * path_loss * self.noise_floor

    def _generate_data(self, path_loss: NDArray, associations: dict) -> dict:
        links = []

        for a, stations in associations.items():
            for s in stations:
                if self._tx_possible(path_loss[a, s]):
                    links.append((f'AP_{a}', f'STA_{s}'))

        problem_data = {
            'stations': [f'STA_{s}' for s in self.stations],
            'access_points': [f'AP_{a}' for a in self.access_points],
            'links': links,
            'link_node_a': {l: l[0] for l in links},  # APs
            'link_node_b': {l: l[1] for l in links},  # STAs
        }

        link_path_loss = {(f'AP_{a}', f'STA_{s}'): path_loss[a, s].item() for a, s in product(self.access_points, self.stations)}
        max_interference = {}

        for l, m in product(links, self.mcs_values):
            max_interference[l, m] = sum(
                self.max_tx_power * (self.min_sinr[m] * link_path_loss[l] / link_path_loss[i, problem_data['link_node_b'][l]]) +
                self.min_sinr[m] * link_path_loss[l] * self.noise_floor
                for i in problem_data['access_points'] if i != problem_data['link_node_a'][l]
            )

        problem_data['link_path_loss'] = link_path_loss
        problem_data['max_interference'] = max_interference
        return problem_data

    def _calculate_station_rates(self, shares: dict, conf_link_rates: dict, link_node_b: dict) -> dict:
        rates = defaultdict(float)

        for t, weight in shares.items():
            if weight <= 0:
                continue

            for link, rate in conf_link_rates[t].items():
                sta = link_node_b[link]
                rates[sta] += weight * rate
        
        return rates

    def _solve_max_min_problem(
        self,
        target_stations: list,
        problem_data: dict,
        configuration: dict,
        sigma: dict,
        rho: dict,
        l_val: float,
        use_delta: bool
    ) -> tuple[dict, dict, float]:
        
        for _ in range(self.max_iterations):
            main_result, main_objective = self.main(
                stations=problem_data['stations'],
                selected_stations=target_stations,
                link_node_b=problem_data['link_node_b'],
                conf_links=configuration['conf_links'],
                conf_link_rates=configuration['conf_link_rates'],
                confs=configuration['confs'],
                sigma=sigma,
                rho=rho,
                l=l_val
            )

            configuration, pricing_objective = self.pricing(
                dual_alpha=main_result['alpha'],
                dual_beta=main_result['beta'],
                dual_gamma=main_result['gamma'],
                dual_delta=main_result['delta'],
                use_delta=use_delta,
                stations=problem_data['stations'],
                access_points=problem_data['access_points'],
                links=problem_data['links'],
                link_node_a=problem_data['link_node_a'],
                link_node_b=problem_data['link_node_b'],
                link_path_loss=problem_data['link_path_loss'],
                max_interference=problem_data['max_interference'],
                configuration=configuration
            )

            if abs(pricing_objective) <= self.epsilon:
                break
        
        return configuration, main_result, pricing_objective

    def __call__(self, path_loss: NDArray, associations: dict) -> tuple[dict, float, list]:
        path_loss = dbm_to_lin(path_loss)
        problem_data = self._generate_data(path_loss, associations)

        l_curr = 0.0
        sigma = {s: 0.0 for s in problem_data['stations']}
        final_configuration = [] 

        configuration = self.pricing.initial_configuration(
            links=problem_data['links'],
            link_path_loss=problem_data['link_path_loss']
        )
        
        active_stations = list(problem_data['stations'])
        pricing_objectives = []

        while active_stations:
            rho = {s: 0.0 for s in problem_data['stations']}

            configuration, global_result, global_objective = self._solve_max_min_problem(
                target_stations=active_stations,
                problem_data=problem_data,
                configuration=configuration,
                sigma=sigma,
                rho=rho,
                l_val=l_curr,
                use_delta=False
            )
            pricing_objectives.append(global_objective)

            rho_global = self._calculate_station_rates(
                global_result['shares'], 
                configuration['conf_link_rates'], 
                problem_data['link_node_b']
            )

            stations_to_remove = []
            
            for s_prime in active_stations:
                configuration, single_result, _ = self._solve_max_min_problem(
                    target_stations=[s_prime],
                    problem_data=problem_data,
                    configuration=configuration,
                    sigma=sigma,
                    rho=rho_global,
                    l_val=l_curr,
                    use_delta=True
                )
                
                rho_single = self._calculate_station_rates(
                    single_result['shares'], 
                    configuration['conf_link_rates'], 
                    problem_data['link_node_b']
                )
                
                if abs(rho_single[s_prime] - rho_global[s_prime]) <= self.epsilon:
                    configurations_to_remove = []
                    
                    for t, weight in single_result['shares'].items():
                        if weight > self.epsilon and s_prime in configuration['conf_links'][t]:
                            for link, rate in configuration['conf_link_rates'][t].items():
                                sta = problem_data['link_node_b'][link]
                                sigma[sta] += weight * rate
                            
                            final_configuration.append({
                                'weight': weight,
                                'links': configuration['conf_links'][t],
                                'rates': configuration['conf_link_rates'][t],
                                'total_rate': configuration['conf_total_rates'][t],
                                'tx_power': configuration['conf_link_tx_power'][t]
                            })
                            l_curr += weight
                            configurations_to_remove.append(t)

                    for t in configurations_to_remove:
                        configuration['confs'].remove(t)
                        del configuration['conf_links'][t]
                        del configuration['conf_link_rates'][t]
                        del configuration['conf_total_rates'][t]
                        del configuration['conf_link_tx_power'][t]

                    stations_to_remove.append(s_prime)
            
            for s in stations_to_remove:
                active_stations.remove(s)

        result = {
            'links': {idx: item['links'] for idx, item in enumerate(final_configuration)},
            'link_rates': {idx: item['rates'] for idx, item in enumerate(final_configuration)},
            'total_rates': {idx: item['total_rate'] for idx, item in enumerate(final_configuration)},
            'tx_power': {idx: {l: lin_to_dbm(p).item() for l, p in item['tx_power'].items()} for idx, item in enumerate(final_configuration)},
            'shares': {idx: item['weight'] for idx, item in enumerate(final_configuration)}
        }

        total_rate = sum(result['total_rates'][c] * result['shares'][c] for c in result['shares'])
        return result, total_rate, pricing_objectives
