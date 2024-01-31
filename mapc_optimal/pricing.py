"""
The pricing algorithm used to propose new configurations for the main problem.
"""

import pulp as plp
from numpy.typing import NDArray


class Pricing:
    r"""
    TODO

    Parameters
    ----------
    mcs_values : list
        List of the available MCS values.
    mcs_data_rates : list
        List of the data rates corresponding to the available MCS values.
    min_sinr : NDArray
        Array containing the minimum SINR values for each MCS value.
    max_tx_power : float
        Maximum transmission power of the nodes.
    min_tx_power : float
        Minimum transmission power of the nodes.
    noise_floor : float
        Mean level of the noise floor in the network.
    opt_sum : bool
        If True, the total throughput is optimized, otherwise the worst throughput is optimized.
    """

    def __init__(
            self,
            mcs_values: list,
            mcs_data_rates: list,
            min_sinr: NDArray,
            max_tx_power: float,
            min_tx_power: float,
            noise_floor: float,
            opt_sum: bool
    ) -> None:
        self.mcs_values = mcs_values
        self.mcs_data_rates = mcs_data_rates
        self.min_sinr = min_sinr
        self.mcs_rate_diff = {m: mcs_data_rates[0] if m == 0 else (mcs_data_rates[m] - mcs_data_rates[m - 1]) for m in mcs_values}
        self.max_tx_power = max_tx_power
        self.min_tx_power = min_tx_power
        self.noise_floor = noise_floor
        self.opt_sum = opt_sum

    def _best_rate(self, path_loss: float) -> float:
        """
        Selects the best data rate for a given node assuming no interference.

        Parameters
        ----------
        path_loss : float
            Path loss between the transmitter and the receiver.

        Returns
        -------
        rate : float
            Maximum possible data rate.
        """

        mcs = (self.max_tx_power >= self.min_sinr * path_loss * self.noise_floor).sum()
        return self.mcs_data_rates[mcs - 1]

    def initial_configuration(self, links: list, link_path_loss: dict) -> dict:
        """
        Generates the initial configuration for the solver. The initial configurations are very simple, they contain
        only one link, with the best possible data rate and the maximum transmission power. The initial configurations
        include all the links in the network.

        Parameters
        ----------
        links : list
            List of the links in the network.
        link_path_loss : dict
            Dictionary containing the path loss of each link.

        Returns
        -------
        configuration : dict
            Dictionary containing the initial configuration.
        """

        configuration = {}
        configuration['confs'] = range(1, len(links) + 1)
        configuration['conf_links'] = {c: [l] for c, l in zip(configuration['confs'], links)}
        configuration['conf_link_rates'] = {c: {} for c in configuration['confs']}
        configuration['conf_link_tx_power'] = {c: {} for c in configuration['confs']}

        for c in configuration['confs']:
            l = configuration['conf_links'][c][0]
            configuration['conf_link_rates'][c][l] = self._best_rate(link_path_loss[l])
            configuration['conf_link_tx_power'][c][l] = self.max_tx_power

        configuration['conf_total_rates'] = {c: sum(configuration['conf_link_rates'][c].values()) for c in configuration['confs']}
        configuration['conf_num'] = len(links) + 1

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
    ) -> tuple[dict, float]:
        """
        TODO

        Parameters
        ----------
        dual_alpha : float
            Dual variable of the alpha constraint.
        dual_beta : dict
            Dual variables of the beta constraints.
        dual_gamma : dict
            Dual variables of the gamma constraints.
        stations : list
            List of the station nodes.
        access_points : list
            List of the access point nodes.
        links : list
            List of the links in the network.
        link_node_a : dict
            Dictionary containing the mapping of links to the access point nodes.
        link_node_b : dict
            Dictionary containing the mapping of links to the station nodes.
        link_path_loss : dict
            Dictionary containing the path loss of each link.
        max_interference : dict
            Dictionary containing the maximum interference level for each link and each MCS value.
        configuration : dict
            Dictionary containing all the configurations.

        Returns
        -------
        result : tuple[dict, float]
            Tuple containing the dictionary with the new configuration and the value of the objective function.
        """

        pricing = plp.LpProblem('pricing', plp.LpMaximize)

        link_tx_power = plp.LpVariable.dicts('link_tx_power', links, lowBound=0, cat=plp.LpContinuous)
        link_on = plp.LpVariable.dicts('link_on', links, cat=plp.LpBinary)
        link_mcs = plp.LpVariable.dicts('link_mcs', [(l, m) for l in links for m in self.mcs_values], cat=plp.LpBinary)
        link_data_rate = plp.LpVariable.dicts('link_data_rate', links, lowBound=0, cat=plp.LpContinuous)
        link_interference = plp.LpVariable.dicts('link_interference', [(l, m) for l in links for m in self.mcs_values], lowBound=0, cat=plp.LpContinuous)

        for s in stations:
            # station receives transmission from at most one AP
            pricing += plp.lpSum(link_on[l] for l in links if link_node_b[l] == s) <= 1, f'station_on_{s}_c'

        for a in access_points:
            # AP can simultaneously transmit to at most one station on all of its links
            pricing += plp.lpSum(link_on[l] for l in links if link_node_a[l] == a) <= 1, f'ap_on_{a}_c'

        for l in links:
            # if link is on, then node can transmit with power constrained by min/max power
            pricing += link_tx_power[l] <= self.max_tx_power * link_on[l], f'link_tx_power_max_{l}_c'
            pricing += link_tx_power[l] >= self.min_tx_power * link_on[l], f'link_tx_power_min_{l}_c'

            for m in self.mcs_values:
                # the way transmission modes are switched on in a link (incremental switching-on)
                if m == 0:
                    pricing += link_mcs[l, 0] <= link_on[l], f'link_mcs_{l}_{m}_c'
                else:
                    pricing += link_mcs[l, m] <= link_mcs[l, m - 1], f'link_mcs_{l}_{m}_c'

                # interference level in link
                pricing += link_interference[l, m] == plp.lpSum(
                    link_tx_power[l_i] * (self.min_sinr[m] * link_path_loss[l] / link_path_loss[link_node_a[l_i], link_node_b[l]]) +
                    self.min_sinr[m] * link_path_loss[l] * self.noise_floor
                    for l_i in links if link_node_a[l_i] != link_node_a[l]
                ), f'link_interference_{l}_{m}_c1'

                # check whether SINR is high enough for transmission with a given MCS
                pricing += link_tx_power[l] + max_interference[l, m] * (1 - link_mcs[l, m]) >= link_interference[l, m], f'link_interference_{l}_{m}_c2'

            # data rate obtained in link (on the basis of the switched-on MCS modes)
            pricing += link_data_rate[l] == plp.lpSum(self.mcs_rate_diff[m] * link_mcs[l, m] for m in self.mcs_values), f'link_data_rate_{l}_c'

        if self.opt_sum:
            # maximization of the total throughput
            pricing += (
                plp.lpSum(link_data_rate[l] for l in links)
                - dual_alpha
                + plp.lpSum(dual_beta[s] * link_data_rate[l] for s in stations for l in links if link_node_b[l] == s)
            ), 'tx_set_throughput_g'
        else:
            # maximization of the worst throughput
            pricing += (
                - dual_alpha
                + plp.lpSum(dual_beta[s] * link_data_rate[l] for s in stations for l in links if link_node_b[l] == s)
            ), 'tx_set_throughput_g'

        pricing.link_on = link_on
        pricing.link_data_rate = link_data_rate
        pricing.link_tx_power = link_tx_power
        pricing.solve()

        if pricing.status != plp.LpStatusOptimal:
            raise Exception('Pricing problem not solved optimally')

        # append the new configuration to the list of configurations
        conf_num = configuration['conf_num']
        configuration['confs'] = range(1, conf_num + 1)
        configuration['conf_links'][conf_num] = [l for l in links if pricing.link_on[l].varValue == 1]
        configuration['conf_link_rates'][conf_num] = {l: pricing.link_data_rate[l].varValue for l in configuration['conf_links'][conf_num]}
        configuration['conf_link_tx_power'][conf_num] = {l: pricing.link_tx_power[l].varValue for l in configuration['conf_links'][conf_num]}
        configuration['conf_total_rates'][conf_num] = sum(configuration['conf_link_rates'][conf_num].values())
        configuration['conf_num'] += 1

        return configuration, plp.value(pricing.objective)
