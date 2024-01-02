from numpy.typing import NDArray

import pulp as plp


class Master:
    def __init__(
            self,
            min_throughput: float,
            opt_sum: bool
    ) -> None:
        self.min_throughput = min_throughput
        self.opt_sum = opt_sum

    def __call__(
            self,
            stations: NDArray,
            link_node_b: dict,
            conf_links: dict,
            conf_link_rates: dict,
            conf_total_rates: dict,
            confs: NDArray
    ) -> tuple:

        # Definition of the master model
        master = plp.LpProblem('master', plp.LpMaximize)

        # Variables:
        # weights of compatible sets (share of time period between configurations)
        conf_weight = plp.LpVariable.dicts('conf_weight', confs, lowBound=0, upBound=1, cat=plp.LpContinuous)

        # throughput of a station
        node_throughput = plp.LpVariable.dicts('node_throughput', stations, lowBound=0, cat=plp.LpContinuous)

        # minimum throughput of all stations
        min_throughput = plp.LpVariable('min_throughput', lowBound=0, cat=plp.LpContinuous)

        # Constraints:
        # percentage of time when we use the configuration
        master += plp.lpSum(conf_weight[c] for c in confs) == 1, 'conf_weight_c'  # dual: alpha

        for s in stations:
            # calculation of the summarized throughput of each station (over all the used configurations)
            master += node_throughput[s] == plp.lpSum(
                conf_link_rates[c, l] * conf_weight[c] for c in confs for l in conf_links[c] if link_node_b[l] == s
            ), f'node_throughput_{s}_c'  # dual: beta

            # calculation of the worst throughput (this constraint is especially necessary for not-opt_sum)
            master += node_throughput[s] >= min_throughput, f'worst_throughput_{s}_c'  # dual: gamma

        if self.opt_sum:
            # enforcement of the worst throughput
            master += min_throughput >= self.min_throughput, f'min_throughput_c'

        # Goal for the master problem
        if self.opt_sum:
            master += plp.lpSum(conf_total_rates[c] * conf_weight[c] for c in confs), 'total_throughput_g'
        else:
            master += min_throughput, 'min_throughput_g'

        # To access the variables from outside
        master.conf_weight = conf_weight

        # Solve the master problem
        master.solve()

        # Check the status of the optimization problem
        if master.status != plp.LpStatusOptimal:
            raise Exception('Master problem not solved optimally')

        # Results with dual variables and shares
        result = {
            'alpha': master.constraints['conf_weight_c'].pi,
            'beta': {s: master.constraints[f'node_throughput_{s}_c'].pi for s in stations},
            'gamma': {s: master.constraints[f'worst_throughput_{s}_c'].pi for s in stations},
            'shares': {c: master.conf_weight[c].varValue for c in confs}
        }

        return result, plp.value(master.objective)
