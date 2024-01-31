"""
The formulation of the main problem solving the selection and division of configurations.
"""

import pulp as plp


class Main:
    r"""
    TODO

    Parameters
    ----------
    min_throughput : float
        The minimum throughput of each station required when optimizing total throughput.
    opt_sum : bool
        If True, the total throughput is optimized, otherwise the worst throughput is optimized.
    """

    def __init__(self, min_throughput: float, opt_sum: bool) -> None:
        self.min_throughput = min_throughput
        self.opt_sum = opt_sum

    def __call__(
            self,
            stations: list,
            link_node_b: dict,
            conf_links: dict,
            conf_link_rates: dict,
            conf_total_rates: dict,
            confs: list
    ) -> tuple[dict, float]:
        """
        TODO

        Parameters
        ----------
        stations : list
            List of the station nodes.
        link_node_b : dict
            Dictionary containing the mapping of links to the station nodes.
        conf_links : dict
            Dictionary containing the links of each configuration.
        conf_link_rates : dict
            Dictionary containing the rates of each link for each configuration.
        conf_total_rates : dict
            Dictionary containing the total rate of each configuration.
        confs : list
            List of the configurations.

        Returns
        -------
        result : tuple[dict, float]
            Tuple containing the results of the optimization problem and the value of the objective function.
        """

        main = plp.LpProblem('main', plp.LpMaximize)

        conf_weight = plp.LpVariable.dicts('conf_weight', confs, lowBound=0, upBound=1, cat=plp.LpContinuous)
        node_throughput = plp.LpVariable.dicts('node_throughput', stations, lowBound=0, cat=plp.LpContinuous)
        min_throughput = plp.LpVariable('min_throughput', lowBound=0, cat=plp.LpContinuous)

        # fraction of time when we use the configuration, the sum of all the weights is 1
        main += plp.lpSum(conf_weight[c] for c in confs) == 1, 'conf_weight_c'  # dual: alpha

        for s in stations:
            # calculation of the summarized throughput of each station (over all the used configurations)
            main += node_throughput[s] == plp.lpSum(
                conf_link_rates[c][l] * conf_weight[c] for c in confs for l in conf_links[c] if link_node_b[l] == s
            ), f'node_throughput_{s}_c'  # dual: beta

            # calculation of the worst throughput (this constraint is especially necessary for not-opt_sum)
            main += node_throughput[s] >= min_throughput, f'worst_throughput_{s}_c'  # dual: gamma

        if self.opt_sum:
            # enforcement of the minimum throughput of each station
            main += min_throughput >= self.min_throughput, f'min_throughput_c'

        if self.opt_sum:
            # maximization of the total throughput
            main += plp.lpSum(conf_total_rates[c] * conf_weight[c] for c in confs), 'total_throughput_g'
        else:
            # maximization of the worst throughput
            main += min_throughput, 'min_throughput_g'

        main.conf_weight = conf_weight
        main.solve()

        if main.status != plp.LpStatusOptimal:
            raise Exception('Main problem not solved optimally')

        result = {
            'alpha': main.constraints['conf_weight_c'].pi,
            'beta': {s: main.constraints[f'node_throughput_{s}_c'].pi for s in stations},
            'gamma': {s: main.constraints[f'worst_throughput_{s}_c'].pi for s in stations},
            'shares': {c: main.conf_weight[c].varValue for c in confs}
        }

        return result, plp.value(main.objective)
