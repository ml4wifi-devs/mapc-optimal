import networkx as nx
import numpy as np
import pulp as plp

from mapc_optimal.utils import lin_to_dbm


class Pricing:
    def __init__(
            self,
            mcs_values: list,
            mcs_data_rates: list,
            min_sinr: np.ndarray,
            max_tx_power: float,
            min_tx_power: float,
            noise_floor: float,
            opt_sum: bool
    ) -> None:
        self.mcs_values = mcs_values
        self.mcs_data_rates = mcs_data_rates
        self.min_sinr = min_sinr
        self.mcs_rate_diff = {m: mcs_data_rates[0].item() if m == 0 else (mcs_data_rates[m] - mcs_data_rates[m - 1]).item() for m in mcs_values}
        self.max_tx_power = max_tx_power
        self.min_tx_power = min_tx_power
        self.noise_floor = noise_floor
        self.opt_sum = opt_sum

    def _best_rate(self, path_loss: float) -> float:
        mcs = (self.max_tx_power >= self.min_sinr * path_loss * self.noise_floor).sum()
        return self.mcs_data_rates[mcs - 1].item()

    def initial_configuration(
            self,
            stations: list,
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
        configuration['conf_link_rates'] = {c: {} for c in configuration['confs']}

        for c in configuration['confs']:
            for l in configuration['conf_links'][c]:
                configuration['conf_link_rates'][c][l] = self._best_rate(link_path_loss[l])

        # total rates for compatible sets
        configuration['conf_total_rates'] = {c: sum(configuration['conf_link_rates'][c].values()) for c in configuration['confs']}

        # tx power for compatible sets
        configuration['conf_link_tx_power'] = {c: {} for c in configuration['confs']}

        for c in configuration['confs']:
            configuration['conf_link_tx_power'][c] = {
                l: lin_to_dbm(self.max_tx_power).item() for l in configuration['conf_links'][c]
            }

        return configuration

    def __call__(
            self,
            dual_alpha: float,
            dual_beta: dict,
            dual_gamma: dict,
            stations: list,
            access_points: list,
            links: list,
            link_node_a: dict,
            link_node_b: dict,
            link_path_loss: dict,
            max_interference: dict,
            configuration: dict
    ) -> tuple:

        # Definition of the pricing problem
        pricing = plp.LpProblem('pricing', plp.LpMaximize)

        # Variables:
        # transmission power used in link
        link_tx_power = plp.LpVariable.dicts('link_tx_power', links, lowBound=0, cat=plp.LpContinuous)

        # which link is on (binary)
        link_on = plp.LpVariable.dicts('link_on', links, cat=plp.LpBinary)

        # MCS mode used in link (binary)
        link_mcs = plp.LpVariable.dicts('link_mcs', [(l, m) for l in links for m in self.mcs_values], cat=plp.LpBinary)

        # data rate obtained in link
        link_data_rate = plp.LpVariable.dicts('link_data_rate', links, lowBound=0, cat=plp.LpContinuous)

        # interference level in link
        link_interference = plp.LpVariable.dicts('link_interference', [(l, m) for l in links for m in self.mcs_values], lowBound=0, cat=plp.LpContinuous)

        # Constraints:
        for s in stations:
            # station receives transmission from at most one AP
            pricing += plp.lpSum(link_on[l] for l in links if link_node_b[l] == s) <= 1, f'station_on_{s}_c'

        for a in access_points:
            # AP can simultaneously transmit to at most one station on all of its links
            pricing += plp.lpSum(link_on[l] for l in links if link_node_a[l] == a) <= 1, f'ap_on_{a}_c'

        for l in links:
            a, s = link_node_a[l], link_node_b[l]

            # if link is on, then node can transmit with power constrained by min/max power
            pricing += link_tx_power[l] <= self.max_tx_power * link_on[l], f'link_tx_power_max_{l}_c'
            pricing += link_tx_power[l] >= self.min_tx_power * link_on[l], f'link_tx_power_min_{l}_c'

            for m in self.mcs_values:
                # the way transmission modes are switched on in link (incremental switching-on)
                if m == 0:
                    pricing += link_mcs[l, 0] <= link_on[l], f'link_mcs_{l}_{m}_c'
                else:
                    pricing += link_mcs[l, m] <= link_mcs[l, m - 1], f'link_mcs_{l}_{m}_c'

                # interference level in link
                pricing += link_interference[l, m] == plp.lpSum(
                    link_tx_power[l_i] * (self.min_sinr[m] * link_path_loss[l] / link_path_loss[link_node_a[l_i], s]) +
                    self.min_sinr[m] * link_path_loss[l] * self.noise_floor
                    for l_i in links if link_node_a[l_i] != a
                ), f'link_interference_{l}_{m}_c1'

                pricing += link_tx_power[l] + max_interference[l, m] * (1 - link_mcs[l, m]) >= link_interference[l, m], f'link_interference_{l}_{m}_c2'

            # data rate obtained in link (on the basis of the switched-on MCS modes)
            pricing += link_data_rate[l] == plp.lpSum(self.mcs_rate_diff[m] * link_mcs[l, m] for m in self.mcs_values), f'link_data_rate_{l}_c'

        # Goal for the pricing problem
        if self.opt_sum:
            pricing += (
                plp.lpSum(link_data_rate[l] for l in links)
                - dual_alpha
                + plp.lpSum(dual_beta[s] * link_data_rate[l] for s in stations for l in links if link_node_b[l] == s)
            ), 'tx_set_throughput_g'
        else:
            pricing += (
                - dual_alpha
                + plp.lpSum(dual_beta[s] * link_data_rate[l] for s in stations for l in links if link_node_b[l] == s)
            ), 'tx_set_throughput_g'

        # To access the variables from outside:
        pricing.link_on = link_on
        pricing.link_data_rate = link_data_rate
        pricing.link_tx_power = link_tx_power

        # Solve the pricing problem
        pricing.solve()

        # Check the status of the optimization problem
        if pricing.status != plp.LpStatusOptimal:
            raise Exception('Pricing problem not solved optimally')

        # Generation of a new compatible set
        conf_num = configuration['conf_num']
        configuration['confs'] = range(1, conf_num + 1)

        # links used in the new compatible set
        configuration['conf_links'][conf_num] = [l for l in links if pricing.link_on[l].varValue == 1]

        # transmission power for the new compatible set
        configuration['conf_link_tx_power'][conf_num] = {
            l: lin_to_dbm(pricing.link_tx_power[l].varValue).item() for l in links if pricing.link_on[l].varValue == 1
        }

        # link rates for the new compatible set
        configuration['conf_link_rates'][conf_num] = {}

        for l in links:
            if pricing.link_on[l].varValue == 1:
                configuration['conf_link_rates'][conf_num][l] = pricing.link_data_rate[l].varValue

        # total rates for compatible sets
        configuration['conf_total_rates'][conf_num] = sum(configuration['conf_link_rates'][conf_num].values())

        # and for the next interation...
        configuration['conf_num'] += 1

        return configuration, plp.value(pricing.objective)
