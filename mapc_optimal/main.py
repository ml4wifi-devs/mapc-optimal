import pulp as plp


class Main:
    def __init__(
            self,
            min_throughput: float,
            opt_sum: bool
    ) -> None:
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
    ) -> tuple:

        # Definition of the main model
        main = plp.LpProblem('main', plp.LpMaximize)

        # Variables:
        # weights of compatible sets (share of time period between configurations)
        conf_weight = plp.LpVariable.dicts('conf_weight', confs, lowBound=0, upBound=1, cat=plp.LpContinuous)

        # throughput of a station
        node_throughput = plp.LpVariable.dicts('node_throughput', stations, lowBound=0, cat=plp.LpContinuous)

        # minimum throughput of all stations
        min_throughput = plp.LpVariable('min_throughput', lowBound=0, cat=plp.LpContinuous)

        # Constraints:
        # percentage of time when we use the configuration
        main += plp.lpSum(conf_weight[c] for c in confs) == 1, 'conf_weight_c'  # dual: alpha

        for s in stations:
            # calculation of the summarized throughput of each station (over all the used configurations)
            main += node_throughput[s] == plp.lpSum(
                conf_link_rates[c][l] * conf_weight[c] for c in confs for l in conf_links[c] if link_node_b[l] == s
            ), f'node_throughput_{s}_c'  # dual: beta

            # calculation of the worst throughput (this constraint is especially necessary for not-opt_sum)
            main += node_throughput[s] >= min_throughput, f'worst_throughput_{s}_c'  # dual: gamma

        if self.opt_sum:
            # enforcement of the worst throughput
            main += min_throughput >= self.min_throughput, f'min_throughput_c'

        # Goal for the main problem
        if self.opt_sum:
            main += plp.lpSum(conf_total_rates[c] * conf_weight[c] for c in confs), 'total_throughput_g'
        else:
            main += min_throughput, 'min_throughput_g'

        # To access the variables from outside
        main.conf_weight = conf_weight

        # Solve the main problem
        main.solve()

        # Check the status of the optimization problem
        if main.status != plp.LpStatusOptimal:
            raise Exception('Main problem not solved optimally')

        # Results with dual variables and shares
        result = {
            'alpha': main.constraints['conf_weight_c'].pi,
            'beta': {s: main.constraints[f'node_throughput_{s}_c'].pi for s in stations},
            'gamma': {s: main.constraints[f'worst_throughput_{s}_c'].pi for s in stations},
            'shares': {c: main.conf_weight[c].varValue for c in confs}
        }

        return result, plp.value(main.objective)
